//! Scalar functions to support queries that perform
//! gap filling.
//!
//!
//! ```sql
//! SELECT
//!   location,
//!   DATE_BIN_GAPFILL(INTERVAL '1 minute', time, '1970-01-01T00:00:00Z') AS minute,
//!   LOCF(AVG(temp))
//!   INTERPOLATE(AVG(humidity))
//! FROM temps
//! WHERE time > NOW() - INTERVAL '6 hours' AND time < NOW()
//! GROUP BY LOCATION, MINUTE
//! ```
//!
//! The functions `DATE_BIN_GAPFILL`, `DATE_BIN_WALLCLOCK_GAPFILL`,
//! `LOCF`, and `INTERPOLATE` are special, in that they don't have
//! normal implementations, but instead are transformed by logical
//! analyzer rule `HandleGapFill` to produce a plan that fills gaps.
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::{
    error::{DataFusionError, Result},
    logical_expr::{ScalarUDF, ScalarUDFImpl, Signature, Volatility},
    physical_plan::ColumnarValue,
};


/// The name of the date_bin_gapfill UDF given to DataFusion.
pub const DATE_BIN_GAPFILL_UDF_NAME: &str = "date_bin_gapfill";


/// Wrapper around date_bin style functions to enable gap filling
/// functionality. Although presented as a scalar function the planner
/// will rewrite queries including this wrapper to use a GapFill node.
#[derive(Debug)]
pub struct GapFillWrapper {
    udf: Arc<ScalarUDF>,
    name: String,
    signature: Signature,
}

impl ScalarUDFImpl for GapFillWrapper {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.udf.inner().return_type(arg_types)
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        Err(DataFusionError::NotImplemented(format!(
            "{} is not yet implemented",
            self.name
        )))
    }
}

impl GapFillWrapper {
    /// Create a new GapFillUDFWrapper around a date_bin style UDF.
    pub fn new(udf: Arc<ScalarUDF>) -> Self {
        let name = format!("{}_gapfill", udf.name());
        // The gapfill UDFs have the same type signature as the underlying UDF.
        let Signature {
            type_signature,
            volatility: _,
        } = udf.signature().clone();
        // We don't want this to be optimized away before we can give a helpful error message
        let signature = Signature {
            type_signature,
            volatility: Volatility::Volatile,
        };
        Self {
            udf,
            name,
            signature,
        }
    }

    /// Get the wrapped UDF.
    pub fn inner(&self) -> &Arc<ScalarUDF> {
        &self.udf
    }
}

/// The name of the locf UDF given to DataFusion.
pub const LOCF_UDF_NAME: &str = "locf";

/// The virtual function definition for the `locf` gap-filling
/// function. This function is never actually invoked, but is used to
/// provider parameters for the GapFill node that is added to the plan.
#[derive(Debug)]
pub struct LocfUDF {
    signature: Signature,
}

impl ScalarUDFImpl for LocfUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        LOCF_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "{LOCF_UDF_NAME} should have at least 1 argument"
            )));
        }
        Ok(arg_types[0].clone())
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        Err(DataFusionError::NotImplemented(format!(
            "{LOCF_UDF_NAME} is not yet implemented"
        )))
    }
}

/// The name of the interpolate UDF given to DataFusion.
pub const INTERPOLATE_UDF_NAME: &str = "interpolate";

/// The virtual function definition for the `interpolate` gap-filling
/// function. This function is never actually invoked, but is used to
/// provider parameters for the GapFill node that is added to the plan.
#[derive(Debug)]
pub struct InterpolateUDF {
    signature: Signature,
}

impl Default for InterpolateUDF {
    fn default() -> Self {
        Self {
            signature: Signature {
                type_signature: vec![DataType::Float64],
                volatility: Volatility::Volatile,
            },
        }
    }
}

impl ScalarUDFImpl for InterpolateUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        INTERPOLATE_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "{INTERPOLATE_UDF_NAME} should have at least 1 argument"
            )));
        }
        Ok(arg_types[0].clone())
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        Err(DataFusionError::NotImplemented(format!(
            "{INTERPOLATE_UDF_NAME} is not yet implemented"
        )))
    }
}
