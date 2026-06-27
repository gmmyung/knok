use knok_core::{AxisSpec, TensorType};

pub(super) fn element_count(ty: &TensorType) -> usize {
    ty.shape.iter().product()
}

pub(super) fn format_shape_list(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn format_usize_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn reassociation_for_rank(rank: usize) -> String {
    let dims = (0..rank)
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[[{dims}]]")
}

pub(super) fn collapse_reassociation_for_removed_axis(rank: usize, axis: usize) -> String {
    if rank <= 1 {
        return reassociation_for_rank(rank);
    }
    let mut groups = Vec::new();
    let mut index = 0;
    while index < rank {
        if index == axis {
            if groups.is_empty() {
                groups.push(vec![index, index + 1]);
                index += 2;
            } else {
                groups.last_mut().expect("group exists").push(index);
                index += 1;
            }
        } else {
            groups.push(vec![index]);
            index += 1;
        }
    }
    format_reassociation_groups(groups)
}

pub(super) fn expand_reassociation_for_inserted_axis(input_rank: usize, axis: usize) -> String {
    let mut groups = Vec::new();
    for input_axis in 0..input_rank {
        if input_axis == axis {
            groups.push(vec![input_axis, input_axis + 1]);
        } else if input_axis < axis {
            groups.push(vec![input_axis]);
        } else {
            groups.push(vec![input_axis + 1]);
        }
    }
    if axis == input_rank {
        if let Some(last) = groups.last_mut() {
            last.push(axis);
        } else {
            groups.push(vec![axis]);
        }
    }
    format_reassociation_groups(groups)
}

fn format_reassociation_groups(groups: Vec<Vec<usize>>) -> String {
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

pub(super) fn format_dim_list(rank: usize) -> String {
    (0..rank)
        .map(|index| format!("d{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn parallel_iterators(rank: usize) -> String {
    (0..rank)
        .map(|_| "\"parallel\"")
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn broadcast_result_type(
    lhs: &TensorType,
    rhs: &TensorType,
) -> anyhow::Result<TensorType> {
    if lhs.elem != rhs.elem {
        anyhow::bail!("binary operands have different element types");
    }
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    Ok(TensorType {
        elem: lhs.elem,
        shape,
    })
}

pub(super) fn ensure_broadcastable(input: &TensorType, output: &TensorType) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    let shape = broadcast_shape(&input.shape, &output.shape)?;
    if shape != output.shape {
        anyhow::bail!(
            "broadcast result shape {:?} does not match requested output {:?}",
            shape,
            output.shape
        );
    }
    Ok(())
}

pub(super) fn axis_broadcast_dimensions(
    input_rank: usize,
    output_rank: usize,
    axis: usize,
) -> anyhow::Result<Vec<usize>> {
    if input_rank + 1 != output_rank {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    if axis >= output_rank {
        anyhow::bail!("axis {axis} is out of bounds for rank {output_rank}");
    }
    Ok(vec![axis])
}

pub(super) fn ensure_axis_broadcastable(
    input: &TensorType,
    output: &TensorType,
    axis: usize,
) -> anyhow::Result<()> {
    if input.elem != output.elem {
        anyhow::bail!("broadcast input and output element types differ");
    }
    if input.rank() + 1 != output.rank() {
        anyhow::bail!("axis broadcast expects exactly one reduced dimension");
    }
    for output_index in 0..output.rank() {
        if output_index == axis {
            continue;
        }
        let input_index = if output_index < axis {
            output_index
        } else {
            output_index - 1
        };
        if input.shape[input_index] != output.shape[output_index] {
            anyhow::bail!(
                "axis broadcast dimension mismatch at output dimension {}: input {} vs output {}",
                output_index,
                input.shape[input_index],
                output.shape[output_index]
            );
        }
    }
    Ok(())
}

pub(super) fn collapse_reassociation_for_squeezed_broadcast(
    input_shape: &[usize],
    aligned_output_shape: &[usize],
) -> String {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pending = Vec::new();
    for (index, (input_dim, output_dim)) in input_shape.iter().zip(aligned_output_shape).enumerate()
    {
        pending.push(index);
        if !(*input_dim == 1 && *output_dim != 1) {
            groups.push(core::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        if let Some(last) = groups.last_mut() {
            last.extend(pending);
        } else {
            groups.push(pending);
        }
    }
    let groups = groups
        .into_iter()
        .map(|group| {
            format!(
                "[{}]",
                group
                    .into_iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{groups}]")
}

pub(super) fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> anyhow::Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_dim = dim_from_trailing(lhs, rank, offset);
        let rhs_dim = dim_from_trailing(rhs, rank, offset);
        let dim = match (lhs_dim, rhs_dim) {
            (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => lhs_dim,
            (Some(1), Some(rhs_dim)) => rhs_dim,
            (Some(lhs_dim), Some(1)) => lhs_dim,
            (None, Some(dim)) | (Some(dim), None) => dim,
            (None, None) => unreachable!("rank is derived from at least one shape"),
            (Some(lhs_dim), Some(rhs_dim)) => {
                anyhow::bail!("broadcast dimension {offset} differs: {lhs_dim} vs {rhs_dim}");
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

fn dim_from_trailing(shape: &[usize], rank: usize, offset: usize) -> Option<usize> {
    let padding = rank - shape.len();
    (offset >= padding).then(|| shape[offset - padding])
}

pub(super) fn reduction_output_shape(
    input_shape: &[usize],
    axis: AxisSpec,
    keep_dims: bool,
) -> Vec<usize> {
    match axis {
        AxisSpec::One(axis) if keep_dims => input_shape
            .iter()
            .enumerate()
            .map(|(index, dim)| if index == axis { 1 } else { *dim })
            .collect(),
        AxisSpec::One(axis) => {
            let mut shape = input_shape.to_vec();
            shape.remove(axis);
            shape
        }
        AxisSpec::All => Vec::new(),
    }
}

pub(super) fn reduction_output_map(input_rank: usize, axis: AxisSpec, keep_dims: bool) -> String {
    match axis {
        AxisSpec::One(axis) if keep_dims => {
            let dims = (0..input_rank)
                .map(|index| {
                    if index == axis {
                        "0".to_string()
                    } else {
                        format!("d{index}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        }
        AxisSpec::One(_) if input_rank == 1 => "()".to_string(),
        AxisSpec::One(axis) => {
            let dims = (0..input_rank)
                .filter(|index| *index != axis)
                .map(|index| format!("d{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            if dims.is_empty() {
                "()".to_string()
            } else {
                format!("({dims})")
            }
        }
        AxisSpec::All => "()".to_string(),
    }
}
