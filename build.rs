//! @file
//! @ingroup GNN_Internal_Logic
#[cfg(feature = "nodejs")]
extern crate napi_build;

fn main() {
    #[cfg(feature = "nodejs")]
    napi_build::setup();
}

