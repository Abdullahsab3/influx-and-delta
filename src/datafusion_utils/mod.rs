

use std::cmp::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};

use datafusion::arrow::array::BooleanArray;
use datafusion::arrow::compute::filter_record_batch;
use datafusion::arrow::datatypes::{DataType, Fields};
use datafusion::common::stats::Precision;
use datafusion::common::{DataFusionError, ToDFSchema};
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::utils::inspect_expr_pre;
use datafusion::logical_expr::{expr::Sort, SortExpr};
use datafusion::physical_expr::execution_props::ExecutionProps;
use datafusion::physical_expr::{create_physical_expr, PhysicalExpr};
use datafusion::physical_optimizer::pruning::PruningPredicate;
use datafusion::physical_plan::{collect, EmptyRecordBatchStream, ExecutionPlan};
use datafusion::prelude::{lit, Column, Expr, SessionContext};
use datafusion::{
    arrow::{
        datatypes::{Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    physical_plan::{RecordBatchStream, SendableRecordBatchStream},
    scalar::ScalarValue,
};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};



/// Helper trait for implementing `partial_cmp` for nested types.
///
/// Example
/// ```rust
/// use std::cmp::Ordering;
/// use datafusion_util::ThenWithOpt;
///
/// // Struct has two fields, lets pretend one can't be compared
/// #[derive(Debug, PartialEq)]
/// struct Foo {
///    a: i32,
///    b: i32, // pretend this can't be compared
///    c: i32,
/// }
///
/// impl PartialOrd for Foo {
///    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///       self.a.partial_cmp(&other.a)
///          // only compare c if a is equal
///          .then_with_opt(|| self.c.partial_cmp(&other.c))
///   }
/// }
///
/// let foo1 = Foo { a: 1, b: 2, c: 3 };
/// let foo2 = Foo { a: 1, b: 4, c: 5 };
/// let foo3 = Foo { a: 1, b: 4, c: 6 };
/// let foo4 = Foo { a: 2, b: 4, c: 6 };
///
/// assert_eq!(foo1.partial_cmp(&foo1), Some(Ordering::Equal));
/// assert_eq!(foo1.partial_cmp(&foo2), Some(Ordering::Less));
/// assert_eq!(foo2.partial_cmp(&foo3), Some(Ordering::Less));
/// assert_eq!(foo4.partial_cmp(&foo1), Some(Ordering::Greater));
/// ```
pub trait ThenWithOpt {
    /// Invoke the closure if the ordering is equal, otherwise return the ordering.
    fn then_with_opt<F: FnOnce() -> Self>(self, f: F) -> Self;
}

impl ThenWithOpt for Option<Ordering> {
    fn then_with_opt<F: FnOnce() -> Self>(self, f: F) -> Self {
        match self {
            Some(Ordering::Equal) => f(),
            other => other,
        }
    }
}

/// Execute the [ExecutionPlan] with a default [SessionContext] and
/// collect the results in memory.
///
/// # Panics
/// If an an error occurs
pub async fn test_collect(plan: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    let session_ctx = SessionContext::new();
    let task_ctx = Arc::new(TaskContext::from(&session_ctx));
    collect(plan, task_ctx).await.unwrap()
}

/// Execute the specified partition of the [ExecutionPlan] with a
/// default [SessionContext] returning the resulting stream.
///
/// # Panics
/// If an an error occurs
pub async fn test_execute_partition(
    plan: Arc<dyn ExecutionPlan>,
    partition: usize,
) -> SendableRecordBatchStream {
    let session_ctx = SessionContext::new();
    let task_ctx = Arc::new(TaskContext::from(&session_ctx));
    plan.execute(partition, task_ctx).unwrap()
}

/// Execute the specified partition of the [ExecutionPlan] with a
/// default [SessionContext] and collect the results in memory.
///
/// # Panics
/// If an an error occurs
pub async fn test_collect_partition(
    plan: Arc<dyn ExecutionPlan>,
    partition: usize,
) -> Vec<RecordBatch> {
    let stream = test_execute_partition(plan, partition).await;
    datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap()
}

/// Filter data from RecordBatch
///
/// Borrowed from DF's <https://github.com/apache/arrow-datafusion/blob/ecd0081bde98e9031b81aa6e9ae2a4f309fcec12/datafusion/src/physical_plan/filter.rs#L186>.
// TODO: if we make DF batch_filter public, we can call that function directly
pub fn batch_filter(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> Result<RecordBatch, DataFusionError> {
    predicate
        .evaluate(batch)
        .and_then(|v| v.into_array(batch.num_rows()))
        .and_then(|array| {
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "Filter predicate evaluated to non-boolean value".to_string(),
                    )
                })
                // apply filter array to record batch
                .and_then(|filter_array| {
                    filter_record_batch(batch, filter_array)
                        .map_err(|err| DataFusionError::ArrowError(err, None))
                })
        })
}

/// Returns a new schema where all the fields are nullable
pub fn nullable_schema(schema: SchemaRef) -> SchemaRef {
    // they are all already nullable
    if schema.fields().iter().all(|f| f.is_nullable()) {
        schema
    } else {
        // make a new schema with all nullable fields
        let new_fields: Fields = schema
            .fields()
            .iter()
            .map(|f| {
                // make a copy of the field, but allow it to be nullable
                f.as_ref().clone().with_nullable(true)
            })
            .collect();

        Arc::new(Schema::new_with_metadata(
            new_fields,
            schema.metadata().clone(),
        ))
    }
}

/// Returns a [`PhysicalExpr`] from the logical [`Expr`] and Arrow [`SchemaRef`]
pub fn create_physical_expr_from_schema(
    props: &ExecutionProps,
    expr: &Expr,
    schema: &SchemaRef,
) -> Result<Arc<dyn PhysicalExpr>, DataFusionError> {
    let df_schema = Arc::clone(schema).to_dfschema_ref()?;
    create_physical_expr(expr, df_schema.as_ref(), props)
}

/// Returns a [`PruningPredicate`] from the logical [`Expr`] and Arrow [`SchemaRef`]
pub fn create_pruning_predicate(
    props: &ExecutionProps,
    expr: &Expr,
    schema: &SchemaRef,
) -> Result<PruningPredicate, DataFusionError> {
    let expr = create_physical_expr_from_schema(props, expr, schema)?;
    PruningPredicate::try_new(expr, Arc::clone(schema))
}


/// Create a timestamp literal for the given UTC nanosecond offset in
/// the timezone specified by [TIME_DATA_TIMEZONE].
///
/// N.B. If [TIME_DATA_TIMEZONE] specifies the None timezone then this
/// function behaves identially to [datafusion::prelude::lit_timestamp_nano].
pub fn lit_timestamptz_nano(ns: i64) -> Expr {
    lit(timestamptz_nano(ns))
}

#[allow(non_snake_case)]
pub fn TIME_DATA_TIMEZONE() -> Option<Arc<str>> {
    None
}

/// Create a scalar timestamp value for the given UTC nanosecond offset
/// in the timezone specified by [TIME_DATA_TIMEZONE].
pub fn timestamptz_nano(ns: i64) -> ScalarValue {
    ScalarValue::TimestampNanosecond(Some(ns), TIME_DATA_TIMEZONE())
}