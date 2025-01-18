use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{execution::SessionState, logical_expr::{LogicalPlan, UserDefinedLogicalNode}, physical_plan::ExecutionPlan, physical_planner::{ExtensionPlanner, PhysicalPlanner}};

use super::{plan_gap_fill, GapFill};

#[derive(Debug, Clone)]
pub struct InfluxDeltaExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for InfluxDeltaExtensionPlanner {
    /// Create a physical plan for an extension node
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> datafusion::error::Result<Option<Arc<dyn ExecutionPlan>>> {
        let any = node.as_any();
        let plan: Option<Arc<dyn ExecutionPlan>> =  if let Some(gap_fill) = any.downcast_ref::<GapFill>() {
            let gap_fill_exec =
                plan_gap_fill(session_state, gap_fill, logical_inputs, physical_inputs)?;
            Some(Arc::new(gap_fill_exec))
        } else {
            None
        };
        Ok(plan)
    }
}

