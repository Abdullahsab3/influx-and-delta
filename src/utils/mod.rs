use std::sync::Arc;

use datafusion::{config::ConfigOptions, execution::{SessionState, SessionStateBuilder}, logical_expr::ScalarUDF, prelude::*};
use deltalake::delta_datafusion::planner::DeltaPlanner;

use crate::{exec::influxdelta_ext_planner::InfluxDeltaExtensionPlanner, query::handle_gapfill::HandleGapFill, query_functions::{date_bin_wallclock::DateBinWallclockUDF, gapfill::{GapFillWrapper, InterpolateUDF}, tz::TzUDF}};
pub const BATCH_SIZE: usize = 8 * 1024;

pub fn session_state() -> SessionState {
    // Enable parquet predicate pushdown optimization
    let mut options = ConfigOptions::new();
    options.execution.parquet.pushdown_filters = true;
    options.execution.parquet.reorder_filters = true;
    options.execution.parquet.schema_force_view_types = false;
    options.execution.time_zone = Some(chrono_tz::Tz::UTC.to_string());
    options.optimizer.repartition_sorts = true;
    options.optimizer.prefer_existing_union = true;
    // ParquetExec now returns estimates rather than actual
    // row counts, so we must use estimates rather than exact to optimize partitioning
    // Related to https://github.com/apache/datafusion/issues/8078
    options
        .execution
        .use_row_number_estimates_to_optimize_partitioning = true;

    let session_cfg = SessionConfig::from(options)
        .with_batch_size(BATCH_SIZE)
        // Tell the datafusion optimizer to avoid repartitioning sorted inputs
        .with_prefer_existing_sort(true)
        .with_parquet_bloom_filter_pruning(true);

    let datafusion_functions = datafusion::functions::all_default_functions();

    let functions = vec![
        Arc::new(ScalarUDF::from(DateBinWallclockUDF::default())),
        Arc::new(ScalarUDF::from(TzUDF::default())),
        Arc::new(ScalarUDF::from(GapFillWrapper::new(datafusion::functions::datetime::date_bin()))),
        Arc::new(ScalarUDF::from(GapFillWrapper::new(Arc::new(ScalarUDF::from(DateBinWallclockUDF::default()))))),
        Arc::new(ScalarUDF::from(InterpolateUDF::default())),
    ];

    SessionStateBuilder::new()
        .with_config(session_cfg)
        .with_default_features()
        .with_scalar_functions(datafusion_functions.into_iter().chain(functions).collect())
        .with_analyzer_rule(Arc::new(HandleGapFill))
        .with_query_planner(Arc::new(DeltaPlanner::<InfluxDeltaExtensionPlanner> {
            extension_planner: InfluxDeltaExtensionPlanner,
        }))
        .build()
}