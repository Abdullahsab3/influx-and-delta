use std::sync::Arc;

use arrow::array::RecordBatch;
use chrono::{DateTime, Utc};
use deltalake::{kernel::{DataType, PrimitiveType, StructField, StructType}, protocol::SaveMode, DeltaOps, DeltaTable};

#[tokio::main]
async fn main() {
    let table = create_table("test_table", "./data").await.unwrap();
    let start_date = "2021-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
    let batch = create_random_readings(start_date);
    DeltaOps(table).write(vec![batch]).await.unwrap();
}


fn schema() -> StructType {
    StructType::new(vec![
        StructField::new(
            "measurement_time".to_string(),
            DataType::Primitive(PrimitiveType::Timestamp),
            false,
        ),
        StructField::new(
            "value".to_string(),
            DataType::Primitive(PrimitiveType::Double),
            false,
        )
    ])
}

fn arrow_schema() -> arrow::datatypes::Schema {
    let fields = vec![
        arrow::datatypes::Field::new("measurement_time", arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into())), false),
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Float64, false),
    ];
    arrow::datatypes::Schema::new(fields)
}


pub async fn create_table(name: &str, location: &str) -> Result<DeltaTable, Box<dyn std::error::Error>> {
    let delta_ops = DeltaOps::try_from_uri(location).await?;
    let table = delta_ops
        .create()
        .with_table_name(name)
        .with_save_mode(SaveMode::Overwrite)
        .with_columns(
            schema()
            .fields()
            .cloned(),
        )
        .await?;
    Ok(table)
}


pub fn create_random_readings(start_date: DateTime<Utc>) -> RecordBatch {
    let mut measurement_time = Vec::new();
    let mut value: Vec<f64> = Vec::new();
    for i in 0..10000 {
        measurement_time.push(start_date + chrono::Duration::seconds((i * 900).into()));
        value.push(rand::random::<f64>());
    }
    RecordBatch::try_new(
        Arc::new(arrow_schema()),
        vec![
            Arc::new(arrow::array::TimestampMicrosecondArray::from(
                measurement_time.iter().map(|measurement_time| measurement_time.timestamp_micros()).collect::<Vec<i64>>()).with_timezone("UTC")),
            Arc::new(arrow::array::Float64Array::from(value)),
        ],
    ).unwrap()
}