//! GPU-accelerated template matching.
//!
//! Faster alternative to [imageproc::template_matching](https://docs.rs/imageproc/latest/imageproc/template_matching/index.html).

#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(unused_variables)]

use std::{borrow::Cow, mem::size_of};
use wgpu::util::DeviceExt;

pub enum MatchTemplateMethod {
    SumOfAbsoluteDifferences,
    SumOfSquaredDifferences,
}

/// Slides a template over the input and scores the match at each point using the requested method.
///
/// This is a shorthand for:
/// ```ignore
/// let matcher = TemplateMatcher::new();
/// matcher.match_template(input, template, method);
/// ```
/// You can use  [find_extremes] to find minimum and maximum values, and their locations in the result image.
pub fn match_template<'a>(
    input: impl Into<Image<'a>>,
    template: impl Into<Image<'a>>,
    method: MatchTemplateMethod,
) -> Image<'static> {
    TemplateMatcher::new().match_template(input, template, method)
}

/// Finds the smallest and largest values and their locations in an image.
pub fn find_extremes(input: &Image<'_>) -> Extremes {
    let mut min_value = f32::MAX;
    let mut min_value_location = (0, 0);
    let mut max_value = f32::MIN;
    let mut max_value_location = (0, 0);

    for y in 0..input.height {
        for x in 0..input.width {
            let idx = (y * input.width) + x;
            let value = input.data[idx as usize];

            if value < min_value {
                min_value = value;
                min_value_location = (x, y);
            }

            if value > max_value {
                max_value = value;
                max_value_location = (x, y);
            }
        }
    }

    Extremes {
        min_value,
        max_value,
        min_value_location,
        max_value_location,
    }
}

pub struct Image<'a> {
    pub data: Cow<'a, [f32]>,
    pub width: u32,
    pub height: u32,
}

impl<'a> Image<'a> {
    pub fn new(data: impl Into<Cow<'a, [f32]>>, width: u32, height: u32) -> Self {
        Self {
            data: data.into(),
            width,
            height,
        }
    }
}

#[cfg(feature = "image")]
impl<'a> From<&'a image::ImageBuffer<image::Luma<f32>, Vec<f32>>> for Image<'a> {
    fn from(img: &'a image::ImageBuffer<image::Luma<f32>, Vec<f32>>) -> Self {
        Self {
            data: Cow::Borrowed(img),
            width: img.width(),
            height: img.height(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Extremes {
    pub min_value: f32,
    pub max_value: f32,
    pub min_value_location: (u32, u32),
    pub max_value_location: (u32, u32),
}

pub struct TemplateMatcher {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl Default for TemplateMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateMatcher {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Adapter request failed")
        });

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .expect("Device request failed")
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/matching.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        Self {
            instance,
            adapter,
            device,
            queue,
            shader,
            pipeline_layout,
            bind_group_layout,
        }
    }

    pub fn match_template<'a>(
        &self,
        input: impl Into<Image<'a>>,
        template: impl Into<Image<'a>>,
        method: MatchTemplateMethod,
    ) -> Image<'static> {
        let input = input.into();
        let template = template.into();

        let entry_point = match method {
            MatchTemplateMethod::SumOfAbsoluteDifferences => "main_sad",
            MatchTemplateMethod::SumOfSquaredDifferences => "main_ssd",
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&self.pipeline_layout),
                module: &self.shader,
                entry_point,
            });

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input_buffer"),
                contents: bytemuck::cast_slice(&input.data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let template_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("template_buffer"),
                contents: bytemuck::cast_slice(&template.data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let result_width = (input.width - template.width + 1) as usize;
        let result_height = (input.height - template.height + 1) as usize;
        let result_size = (result_width * result_height) as u64 * size_of::<f32>() as u64;

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result_buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            size: result_size,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            size: result_size,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct Uniforms {
            input_width: u32,
            input_height: u32,
            template_width: u32,
            template_height: u32,
        }

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniform_buffer"),
                contents: bytemuck::cast_slice(&[Uniforms {
                    input_width: input.width,
                    input_height: input.height,
                    template_width: template.width,
                    template_height: template.height,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: template_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (result_width as f32 / 16.0).ceil() as u32,
                (result_height as f32 / 16.0).ceil() as u32,
                1,
            );
        }

        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(async {
            let result;

            if let Some(Ok(())) = receiver.receive().await {
                let data = buffer_slice.get_mapped_range();
                result = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
            } else {
                result = vec![0.0; result_width * result_height]
            };

            Image::new(result, result_width as _, result_height as _)
        })
    }
}
