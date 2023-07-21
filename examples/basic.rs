use std::time::Instant;

use image::{DynamicImage, GenericImageView};
use template_matching::{find_extremes, MatchTemplateMethod, TemplateMatcher};

fn main() {
    let input_image = image::load_from_memory(include_bytes!("ferris.png")).unwrap();
    let input_luma8 = input_image.to_luma8();
    let input_luma32f = input_image.to_luma32f();

    let mut matcher = TemplateMatcher::new();

    for i in 0..5 {
        let n = 10 + i * 5;
        let template_image = DynamicImage::ImageRgba8(input_image.view(n, n, n, n).to_image());
        let template_luma8 = template_image.to_luma8();
        let template_luma32f = template_image.to_luma32f();

        let time = Instant::now();
        let result = matcher.match_template(
            &input_luma32f,
            &template_luma32f,
            MatchTemplateMethod::SumOfSquaredDifferences,
        );
        println!(
            "template_matching::match_template took {} ms",
            time.elapsed().as_millis()
        );

        let extremes = find_extremes(&result);
        println!("{:?}", extremes);

        let time = Instant::now();
        let result = imageproc::template_matching::match_template(
            &input_luma8,
            &template_luma8,
            imageproc::template_matching::MatchTemplateMethod::SumOfSquaredErrors,
        );
        println!(
            "imageproc::template_matching::match_template took {} ms",
            time.elapsed().as_millis()
        );

        let extremes = imageproc::template_matching::find_extremes(&result);
        println!("{:?}", extremes);
        println!();
    }
}
