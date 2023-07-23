# template-matching

[![Latest version](https://img.shields.io/crates/v/template-matching.svg)](https://crates.io/crates/template-matching)
[![Documentation](https://docs.rs/template-matching/badge.svg)](https://docs.rs/template-matching)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

GPU-accelerated template matching library for Rust. The crate is designed as a faster alternative to [imageproc::template_matching](https://docs.rs/imageproc/latest/imageproc/template_matching/index.html).

## Installation

```bash
[dependencies]
template-matching = { version = "0.1.0", features = ["image"] }
```

## Usage

```rust
use template_matching::{find_extremes, match_template, MatchTemplateMethod, TemplateMatcher};

fn main() {
    // Load images and convert them to f32 grayscale
    let input_image = image::load_from_memory(include_bytes!("input.png")).unwrap().to_luma32f();
    let template_image = image::load_from_memory(include_bytes!("template.png")).unwrap().to_luma32f();

    let result = match_template(&input_image, &template_image, MatchTemplateMethod::SumOfSquaredDifferences);

    // Or alternatively you can create the matcher first
    let mut matcher = TemplateMatcher::new();
    matcher.match_template(&input_image, &template_image, MatchTemplateMethod::SumOfSquaredDifferences);
    let result = matcher.wait_for_result().unwrap();

    // Calculate min & max values
    let extremes = find_extremes(&result);
}
```
