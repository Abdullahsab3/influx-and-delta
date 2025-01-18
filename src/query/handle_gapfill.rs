//! An optimizer rule that transforms a plan
//! to fill gaps in time series data.
pub mod range_predicate;
mod virtual_function;



use datafusion::common::{internal_datafusion_err, plan_datafusion_err, plan_err, DFSchema};
use datafusion::logical_expr::{expr::AggregateFunction, ExprSchemable};
use datafusion::scalar::ScalarValue;
use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion, TreeNodeRewriter},
    config::ConfigOptions,
    error::{DataFusionError, Result},
    logical_expr::{
        expr::{Alias, ScalarFunction},
        utils::expr_to_columns,
        Aggregate, Extension, LogicalPlan, Projection, ScalarUDF,
    },
    optimizer::AnalyzerRule,
    prelude::{col, Column, Expr},
};
use hashbrown::{hash_map, HashMap};
use crate::exec::{FillStrategy, GapFill, GapFillParams};
use crate::query_functions::gapfill::{GapFillWrapper, INTERPOLATE_UDF_NAME, LOCF_UDF_NAME};
use crate::query_functions::tz::TzUDF;
use std::{
    collections::HashSet,
    ops::{Bound, Range},
    sync::Arc,
};
use virtual_function::{VirtualFunction, VirtualFunctionFinder};

/// This optimizer rule enables gap-filling semantics for SQL queries
/// that contain calls to `DATE_BIN_GAPFILL()` and related functions
/// like `LOCF()`.
///
/// In SQL a typical gap-filling query might look like this:
/// ```sql
/// SELECT
///   location,
///   DATE_BIN_GAPFILL(INTERVAL '1 minute', time, '1970-01-01T00:00:00Z') AS minute,
///   LOCF(AVG(temp))
/// FROM temps
/// WHERE time > NOW() - INTERVAL '6 hours' AND time < NOW()
/// GROUP BY LOCATION, MINUTE
/// ```
///
/// The initial logical plan will look like this:
///
/// ```text
///   Projection: location, date_bin_gapfill(...) as minute, LOCF(AVG(temps.temp))
///     Aggregate: groupBy=[[location, date_bin_gapfill(...)]], aggr=[[AVG(temps.temp)]]
///       ...
/// ```
///
/// This optimizer rule transforms it to this:
///
/// ```text
///   Projection: location, date_bin_gapfill(...) as minute, AVG(temps.temp)
///     GapFill: groupBy=[[location, date_bin_gapfill(...))]], aggr=[[LOCF(AVG(temps.temp))]], start=..., stop=...
///       Aggregate: groupBy=[[location, date_bin(...))]], aggr=[[AVG(temps.temp)]]
///         ...
/// ```
///
/// For `Aggregate` nodes that contain calls to `DATE_BIN_GAPFILL`, this rule will:
/// - Convert `DATE_BIN_GAPFILL()` to `DATE_BIN()`
/// - Create a `GapFill` node that fills in gaps in the query
/// - The range for gap filling is found by analyzing any preceding `Filter` nodes
///
/// If there is a `Projection` above the `GapFill` node that gets created:
/// - Look for calls to gap-filling functions like `LOCF`
/// - Push down these functions into the `GapFill` node, updating the fill strategy for the column.
///
/// Note: both `DATE_BIN_GAPFILL` and `LOCF` are functions that don't have implementations.
/// This rule must rewrite the plan to get rid of them.
#[derive(Debug)]
pub struct HandleGapFill;

impl HandleGapFill {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for HandleGapFill {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyzerRule for HandleGapFill {
    fn name(&self) -> &str {
        "handle_gap_fill"
    }
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up(handle_gap_fill).map(|t| t.data)
    }
}

fn handle_gap_fill(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let res = match plan {
        LogicalPlan::Aggregate(aggr) => {
            handle_aggregate(aggr).map_err(|e| e.context("handle_aggregate"))?
        }
        LogicalPlan::Projection(proj) => {
            handle_projection(proj).map_err(|e| e.context("handle_projection"))?
        }
        _ => Transformed::no(plan),
    };

    if !res.transformed {
        // no transformation was applied,
        // so make sure the plan is not using gap filling
        // functions in an unsupported way.
        check_node(&res.data)?;
    }

    Ok(res)
}

fn handle_aggregate(aggr: Aggregate) -> Result<Transformed<LogicalPlan>> {
    match replace_date_bin_gapfill(aggr).map_err(|e| e.context("replace_date_bin_gapfill"))? {
        // No change: return as-is
        RewriteInfo::Unchanged(aggr) => Ok(Transformed::no(LogicalPlan::Aggregate(aggr))),
        // Changed: new_aggr has DATE_BIN_GAPFILL replaced with DATE_BIN.
        RewriteInfo::Changed {
            new_aggr,
            date_bin_gapfill_index,
            date_bin_gapfill_args,
            date_bin_udf,
        } => {
            let new_aggr_plan = LogicalPlan::Aggregate(new_aggr);
            check_node(&new_aggr_plan).map_err(|e| e.context("check_node"))?;

            let new_gap_fill_plan = build_gapfill_node(
                new_aggr_plan,
                date_bin_gapfill_index,
                date_bin_gapfill_args,
                date_bin_udf,
            )
            .map_err(|e| e.context("build_gapfill_node"))?;
            Ok(Transformed::yes(new_gap_fill_plan))
        }
    }
}

fn build_gapfill_node(
    new_aggr_plan: LogicalPlan,
    date_bin_gapfill_index: usize,
    date_bin_gapfill_args: Vec<Expr>,
    date_bin_udf: Arc<str>,
) -> Result<LogicalPlan> {
    match date_bin_gapfill_args.len() {
        2 | 3 => (),
        nargs @ (0 | 1 | 4..) => {
            return Err(DataFusionError::Plan(format!(
                "DATE_BIN_GAPFILL expects 2 or 3 arguments, got {nargs}",
            )));
        }
    }

    let mut args_iter = date_bin_gapfill_args.into_iter();

    // Ensure that stride argument is a scalar
    let stride = args_iter.next().unwrap();
    validate_scalar_expr("stride argument to DATE_BIN_GAPFILL", &stride)
        .map_err(|e| e.context("validate_scalar_expr"))?;

    fn get_column(expr: Expr) -> Result<Column> {
        match expr {
            Expr::Column(c) => Ok(c),
            Expr::Cast(c) => get_column(*c.expr),
            Expr::ScalarFunction(ScalarFunction { func, args })
                if func.inner().as_any().is::<TzUDF>() =>
            {
                get_column(args[0].clone())
            }
            _ => Err(DataFusionError::Plan(
                "DATE_BIN_GAPFILL requires a column as the source argument".to_string(),
            )),
        }
    }

    // Ensure that the source argument is a column
    let time_col =
        get_column(args_iter.next().unwrap()).map_err(|e| e.context("get time column"))?;

    // Ensure that a time range was specified and is valid for gap filling
    let time_range = range_predicate::find_time_range(new_aggr_plan.inputs()[0], &time_col)
        .map_err(|e| e.context("find time range"))?;
    validate_time_range(&time_range).map_err(|e| e.context("validate time range"))?;

    // Ensure that origin argument is a scalar
    let origin = args_iter.next();
    if let Some(ref origin) = origin {
        validate_scalar_expr("origin argument to DATE_BIN_GAPFILL", origin)
            .map_err(|e| e.context("validate origin"))?;
    }

    // Make sure the time output to the gapfill node matches what the
    // aggregate output was.
    let time_column = col(datafusion::common::Column::from(
        new_aggr_plan
            .schema()
            .qualified_field(date_bin_gapfill_index),
    ));

    let LogicalPlan::Aggregate(aggr) = &new_aggr_plan else {
        return Err(DataFusionError::Internal(format!(
            "Expected Aggregate plan, got {}",
            new_aggr_plan.display()
        )));
    };
    let mut new_group_expr: Vec<_> = aggr
        .schema
        .iter()
        .map(|(qualifier, field)| {
            Expr::Column(datafusion::common::Column::from((
                qualifier,
                field.as_ref(),
            )))
        })
        .collect();
    let aggr_expr = new_group_expr.split_off(aggr.group_expr.len());

    match (aggr_expr.len(), aggr.aggr_expr.len()) {
        (f, e) if f != e => return Err(internal_datafusion_err!(
            "The number of aggregate expressions has gotten lost; expected {e}, found {f}. This is a bug, please report it."
        )),
        _ => ()
    }

    // this schema is used for the `FillStrategy::Default` checks below. It also represents the
    // schema of the projection of `aggr`, meaning that it shows the columns/fields as they exist
    // after they've been transformed by `aggr` and once they're being input into the next step.
    // Because we are using it to get the data types of the output columns, to then get the default
    // value of those types according to the AggregateFunction below, it all works out.
    let schema = &aggr.schema;

    let fill_behavior = aggr_expr
        .iter()
        .cloned()
        // `aggr_expr` and `aggr.aggr_expr` should line up in the sense that `aggr.aggr_expr[n]`
        // represents a transformation that was done to produce `aggr_expr[n]`, so we can zip them
        // together like this to determine the correct fill type for the produced expression
        .zip(aggr.aggr_expr.iter())
        .map(|(col_expr, aggr_expr)| {
            // if aggr_expr is a function, then it may need special handling for what its
            // FillStrategy may be - specifically, it may produce a non-nullable column, so we need
            // to check if that's the case, and if it is, we need to determine the correct default
            // type to fill in with gapfill instead of just placing nulls. We also need to verify
            // that, after it's passed through `Aggregate`, it does produce a column - since
            // `col_expr` should be the 'computed'/'transformed' representation of `aggr_expr`, we
            // `aggr_expr`, we need to make sure that it's a column or else this doesn't really
            // matter to calculate.
            default_return_value_for_aggr_fn(aggr_expr, schema, col_expr.try_as_col())
                .map(|rt| (col_expr, FillStrategy::Default(rt)))
        })
        .collect::<Result<_>>()?;

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(
            GapFill::try_new(
                Arc::new(new_aggr_plan),
                new_group_expr,
                aggr_expr,
                GapFillParams {
                    date_bin_udf,
                    stride,
                    time_column,
                    origin,
                    time_range,
                    fill_strategy: fill_behavior,
                },
            )
            .map_err(|e| e.context("GapFill::try_new"))?,
        ),
    }))
}

fn validate_time_range(range: &Range<Bound<Expr>>) -> Result<()> {
    let Range { ref start, ref end } = range;
    let (start, end) = match (start, end) {
        (Bound::Unbounded, Bound::Unbounded) => {
            return Err(DataFusionError::Plan(
                "gap-filling query is missing both upper and lower time bounds".to_string(),
            ))
        }
        (Bound::Unbounded, _) => Err(DataFusionError::Plan(
            "gap-filling query is missing lower time bound".to_string(),
        )),
        (_, Bound::Unbounded) => Err(DataFusionError::Plan(
            "gap-filling query is missing upper time bound".to_string(),
        )),
        (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) => Ok((start, end)),
    }?;
    validate_scalar_expr("lower time bound", start)?;
    validate_scalar_expr("upper time bound", end)
}

fn validate_scalar_expr(what: &str, e: &Expr) -> Result<()> {
    let mut cols = HashSet::new();
    expr_to_columns(e, &mut cols)?;
    if !cols.is_empty() {
        Err(DataFusionError::Plan(format!(
            "{what} for gap fill query must evaluate to a scalar"
        )))
    } else {
        Ok(())
    }
}

enum RewriteInfo {
    // Group expressions were unchanged
    Unchanged(Aggregate),
    // Group expressions were changed
    Changed {
        // Group expressions with DATE_BIN_GAPFILL rewritten to DATE_BIN.
        new_aggr: Aggregate,
        // The index of the group expression that contained the call to DATE_BIN_GAPFILL.
        date_bin_gapfill_index: usize,
        // The arguments to the call to DATE_BIN_GAPFILL.
        date_bin_gapfill_args: Vec<Expr>,
        // The name of the UDF that provides the DATE_BIN like functionality.
        date_bin_udf: Arc<str>,
    },
}

// Iterate over the group expression list.
// If it finds no occurrences of date_bin_gapfill, it will return None.
// If it finds more than one occurrence it will return an error.
// Otherwise it will return a RewriteInfo for the analyzer rule to use.
fn replace_date_bin_gapfill(aggr: Aggregate) -> Result<RewriteInfo> {
    let mut date_bin_gapfill_count = 0;
    let mut dbg_idx = None;
    aggr.group_expr
        .iter()
        .enumerate()
        .try_for_each(|(i, e)| -> Result<()> {
            let mut functions = vec![];
            e.visit(&mut VirtualFunctionFinder::new(&mut functions))?;
            let fn_cnt = functions
                .iter()
                .filter(|vf| matches!(vf, VirtualFunction::GapFill(_)))
                .count();
            date_bin_gapfill_count += fn_cnt;
            if fn_cnt > 0 {
                dbg_idx = Some(i);
            }
            Ok(())
        })?;

    let (date_bin_gapfill_index, date_bin) = match date_bin_gapfill_count {
        0 => return Ok(RewriteInfo::Unchanged(aggr)),
        1 => {
            // Make sure that the call to DATE_BIN_GAPFILL is root expression
            // excluding aliases.
            let dbg_idx = dbg_idx.expect("should be found exactly one call");
            VirtualFunction::maybe_from_expr(unwrap_alias(&aggr.group_expr[dbg_idx]))
                .and_then(|vf| vf.date_bin_udf().cloned())
                .map(|f| (dbg_idx, f))
                .ok_or(plan_datafusion_err!("DATE_BIN_GAPFILL must be a top-level expression in the GROUP BY clause when gap filling. It cannot be part of another expression or cast"))?
        }
        _ => {
            return Err(DataFusionError::Plan(
                "DATE_BIN_GAPFILL specified more than once".to_string(),
            ))
        }
    };

    let date_bin_udf = Arc::from(date_bin.name());
    let mut rewriter = DateBinGapfillRewriter {
        args: None,
        date_bin,
    };
    let new_group_expr = aggr
        .group_expr
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            if i == date_bin_gapfill_index {
                e.rewrite(&mut rewriter).map(|t| t.data)
            } else {
                Ok(e)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let date_bin_gapfill_args = rewriter.args.expect("should have found args");

    // Create the aggregate node with the same output schema as the original
    // one. This means that there will be an output column called `date_bin_gapfill(...)`
    // even though the actual expression populating that column will be `date_bin(...)`.
    // This seems acceptable since it avoids having to deal with renaming downstream.
    let new_aggr =
        Aggregate::try_new_with_schema(aggr.input, new_group_expr, aggr.aggr_expr, aggr.schema)
            .map_err(|e| e.context("Aggregate::try_new_with_schema"))?;
    Ok(RewriteInfo::Changed {
        new_aggr,
        date_bin_gapfill_index,
        date_bin_gapfill_args,
        date_bin_udf,
    })
}

fn unwrap_alias(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Alias(Alias { expr, .. }) => e = expr.as_ref(),
            e => break e,
        }
    }
}

struct DateBinGapfillRewriter {
    args: Option<Vec<Expr>>,
    date_bin: Arc<ScalarUDF>,
}

impl TreeNodeRewriter for DateBinGapfillRewriter {
    type Node = Expr;
    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match &expr {
            Expr::ScalarFunction(fun) if fun.func.inner().as_any().is::<GapFillWrapper>() => {
                Ok(Transformed::new(expr, true, TreeNodeRecursion::Jump))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }

    fn f_up(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        // We need to preserve the name of the original expression
        // so that everything stays wired up.
        let orig_name = expr.schema_name().to_string();
        match expr {
            Expr::ScalarFunction(ScalarFunction { func, args })
                if func.inner().as_any().is::<GapFillWrapper>() =>
            {
                self.args = Some(args.clone());
                Ok(Transformed::yes(
                    Expr::ScalarFunction(ScalarFunction {
                        func: Arc::clone(&self.date_bin),
                        args,
                    })
                    .alias(orig_name),
                ))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }
}

fn udf_to_fill_strategy(name: &str) -> Option<FillStrategy> {
    match name {
        LOCF_UDF_NAME => Some(FillStrategy::PrevNullAsMissing),
        INTERPOLATE_UDF_NAME => Some(FillStrategy::LinearInterpolate),
        _ => None,
    }
}

fn handle_projection(mut proj: Projection) -> Result<Transformed<LogicalPlan>> {
    let Some(child_gapfill) = (match proj.input.as_ref() {
        LogicalPlan::Extension(Extension { node }) => node.as_any().downcast_ref::<GapFill>(),
        _ => None,
    }) else {
        // If this is not a projection that is a parent to a GapFill node,
        // then there is nothing to do.
        return Ok(Transformed::no(LogicalPlan::Projection(proj)));
    };

    let mut fill_fn_rewriter = FillFnRewriter {
        aggr_col_fill_map: HashMap::new(),
    };

    let new_proj_exprs = proj
        .expr
        .iter()
        .map(|expr| {
            expr.clone()
                .rewrite(&mut fill_fn_rewriter)
                .map(|t| t.data)
                .map_err(|e| e.context(format!("rewrite: {expr}")))
        })
        .collect::<Result<Vec<Expr>>>()?;
    let FillFnRewriter { aggr_col_fill_map } = fill_fn_rewriter;

    if aggr_col_fill_map.is_empty() {
        return Ok(Transformed::no(LogicalPlan::Projection(proj)));
    }

    // Clone the existing GapFill node, then modify it in place
    // to reflect the new fill strategy.
    let mut new_gapfill = child_gapfill.clone();
    for (e, (fs, udf)) in aggr_col_fill_map {
        if new_gapfill.replace_fill_strategy(&e, fs).is_none() {
            // There was a gap filling function called on a non-aggregate column.
            return Err(DataFusionError::Plan(format!(
                "{udf} must be called on an aggregate column in a gap-filling query",
            )));
        }
    }
    proj.expr = new_proj_exprs;
    proj.input = Arc::new(LogicalPlan::Extension(Extension {
        node: Arc::new(new_gapfill),
    }));
    Ok(Transformed::yes(LogicalPlan::Projection(proj)))
}

/// Implements `TreeNodeRewriter`:
/// - Traverses over the expressions in a projection node
/// - If it finds a function that requires a non-Null FillStrategy (determined by
///   `udf_to_fill_strategy`), it replaces them with `col AS <original name>`
/// - Collects into [`Self::aggr_col_fill_map`] which correlates
///   aggregate columns to their [`FillStrategy`].
struct FillFnRewriter {
    aggr_col_fill_map: HashMap<Expr, (FillStrategy, String)>,
}

impl TreeNodeRewriter for FillFnRewriter {
    type Node = Expr;
    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match &expr {
            Expr::ScalarFunction(fun) if udf_to_fill_strategy(fun.name()).is_some() => {
                Ok(Transformed::new(expr, true, TreeNodeRecursion::Jump))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }

    fn f_up(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        let orig_name = expr.schema_name().to_string();
        match expr {
            Expr::ScalarFunction(mut fun) => {
                let Some(fs) = udf_to_fill_strategy(fun.name()) else {
                    return Ok(Transformed::no(Expr::ScalarFunction(fun)));
                };

                let arg = fun.args.remove(0);
                self.add_fill_strategy(arg.clone(), fs, fun.name().to_string())?;
                Ok(Transformed::yes(arg.alias(orig_name)))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }
}

impl FillFnRewriter {
    fn add_fill_strategy(&mut self, e: Expr, fs: FillStrategy, name: String) -> Result<()> {
        match self.aggr_col_fill_map.entry(e) {
            hash_map::Entry::Occupied(_) => Err(DataFusionError::NotImplemented(
                "multiple fill strategies for the same column".to_string(),
            )),
            hash_map::Entry::Vacant(ve) => {
                ve.insert((fs, name));
                Ok(())
            }
        }
    }
}

fn check_node(node: &LogicalPlan) -> Result<()> {
    node.expressions().iter().try_for_each(|expr| {
        let mut functions = vec![];
        expr.visit(&mut VirtualFunctionFinder::new(&mut functions))?;
        if functions.is_empty() {
            Ok(())
        } else {
            // There should be no virtual functions in this node, base the error message on the first one.
            match &functions[0] {
                VirtualFunction::GapFill(wrapped) => plan_err!(
                    "{}_gapfill may only be used as a GROUP BY expression",
                    wrapped.name()
                ),
                VirtualFunction::Locf => plan_err!("{LOCF_UDF_NAME} may only be used in the SELECT list of a gap-filling query"),
                VirtualFunction::Interpolate=> plan_err!("{INTERPOLATE_UDF_NAME} may only be used in the SELECT list of a gap-filling query"),
            }
        }
    })
}

/// Tries to process `maybe_wrapped_fun` as an [`AggregateFunction`] which may be wrapped by a type
/// which does not change its return type (e.g. [`Alias`]; read comment on `get_aggr_fn` inside for
/// more detail) and get its default return value, assuming that its output column (if `Some(_)` is
/// passed in for `output_column`) or its arguments (if `None` is passed in for `output_column`)
/// exist in `schema`.
///
/// This also only returns `Ok(Some(_))` if the wrapped func produces non-nullable output - because
/// this function, at time of writing, is only used to facilitate gapfilling values for
/// non-nullable columns, we only care about the return types for functions that cannot have
/// nullable outputs
///
/// # Arguments
///
/// * `maybe_wrapped_fun` - An expression that may contain an `AggregateFunction` whose output type
///   will not be transformed by the expressions wrapping it. If this turns out to not contain such a
///   AggregateFunction, this function returns Ok(None)
/// * `schema` - The schema that the output column or agg func argument types are expected to
///   reside in. read comment above for meaning of 'or' here
/// * `output_column` - A [`Column`] representing the output of this AggregateFunction with this
///   schema, if it is already know. `None` can be passed in for this function if unknown; passing in
///   `Some` just unlocks small performance gains.
pub fn default_return_value_for_aggr_fn(
    maybe_wrapped_fun: &Expr,
    schema: &DFSchema,
    output_column: Option<&Column>,
) -> Result<ScalarValue> {
    // we use this function to recurse through the structure of the expr that we're given to try to
    // find an aggregate function nested deep in it which still retains its same return type. E.g.
    // if an aggregate function is nested inside a `between`, it doesn't retain its return type,
    // since the type of the column is the return type of `between`, not of the function. But if
    // it's inside an `Alias`, it does retain its return type, since the output is not transformed
    // at all (just labeled differently).
    fn get_aggr_fn(expr: &Expr) -> Option<&AggregateFunction> {
        match expr {
            Expr::AggregateFunction(ref fun) => Some(fun),
            Expr::Alias(alias) => get_aggr_fn(&alias.expr),
            _ => None,
        }
    }

    // If there's not an aggregate function in there, we don't care
    let Some(fun) = get_aggr_fn(maybe_wrapped_fun) else {
        return maybe_wrapped_fun.get_type(schema)?.try_into();
    };

    // so if we already know the output column...
    output_column
        .map(|col| {
            // ...just get its type.
            schema
                .field_from_column(col)
                .map(|field| field.data_type().clone())
        })
        .unwrap_or_else(|| {
            // if we don't know the output column, query the aggregate function for its return type
            // with the provided arguments
            let args = fun
                .args
                .iter()
                .map(|arg| arg.get_type(schema))
                .collect::<Result<Vec<_>, _>>()?;

            fun.func.return_type(&args)
        })
        // and then get the default value for that return type.
        .and_then(|return_type| fun.func.default_value(&return_type))
}