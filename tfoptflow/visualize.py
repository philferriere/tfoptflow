"""
visualize.py

Visualization helpers.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt

from utils import clean_dst_file
from optflow import flow_to_img


def plot_img_pairs_w_flows(
        img_pairs,
        flow_pyrs=None,
        num_lvls=0,
        flow_preds=None,
        flow_gts=None,
        titles=None,
        info=None,
        flow_mag_max=None):
    """Plot the given set of image pairs, optionally with flows and titles.
    Args:
        img_pairs: image pairs in [batch_size, 2, H, W, 3] or list([2, H, W, 3]) format.
        flow_pyrs: optional, predicted optical flow pyramids [batch_size, H, W, 2] or list([H, W, 2]) format.
        num_lvls: number of levels to show per pyramid (flow_pyrs must be set)
        flow_preds: optional, predicted flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        flow_gts: optional, groundtruth flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        titles: optional, list of image and flow IDs to display with each image.
        info: optional, stats to display above predicted flow
        flow_mag_max: Max flow to map to 255
    Returns:
        plt: plot
    """
    # Setup drawing canvas
    fig_height, fig_width = 5, 5
    row_count = len(img_pairs)
    col_count = 2
    if flow_preds is not None:
        col_count += 1
    if flow_gts is not None:
        col_count += 1
    if flow_pyrs is not None:
        row_count += len(img_pairs)
        jump = num_lvls - col_count
        col_count = max(num_lvls, col_count)
    plt.figure(figsize=(fig_width * col_count, fig_height * row_count))

    # Plot img_pairs inside the canvas
    plot = 1
    for row in range(len(img_pairs)):
        # Plot image pair
        plt.subplot(row_count, col_count, plot)
        if titles is not None:
            plt.title(titles[row][0], fontsize=fig_width * 2)
        plt.axis('off')
        plt.imshow(img_pairs[row][0])
        plt.subplot(row_count, col_count, plot + 1)
        if titles is not None:
            plt.title(titles[row][1], fontsize=fig_width * 2)
        plt.axis('off')
        plt.imshow(img_pairs[row][1])
        plot += 2

        # Plot predicted flow, if any
        if flow_preds is not None:
            plt.subplot(row_count, col_count, plot)
            title = "predicted flow " + info[row] if info is not None else "predicted flow"
            plt.title(title, fontsize=fig_width * 2)
            plt.axis('off')
            plt.imshow(flow_to_img(flow_preds[row], flow_mag_max=flow_mag_max))
            plot += 1

        # Plot groundtruth flow, if any
        if flow_gts is not None:
            plt.subplot(row_count, col_count, plot)
            plt.title("groundtruth flow", fontsize=fig_width * 2)
            plt.axis('off')
            plt.imshow(flow_to_img(flow_gts[row], flow_mag_max=flow_mag_max))
            plot += 1

        # Plot the flow pyramid on the next row
        if flow_pyrs is not None:
            if jump > 0:
                plot += jump
            for lvl in range(num_lvls):
                plt.subplot(row_count, col_count, plot)
                plt.title(f"level {len(flow_pyrs[row]) - lvl + 1}", fontsize=fig_width * 2)
                plt.axis('off')
                plt.imshow(flow_to_img(flow_pyrs[row][lvl], flow_mag_max=flow_mag_max))
                plot += 1
            if jump < 0:
                plot -= jump

    plt.tight_layout()
    return plt


def display_img_pairs_w_flows(
        img_pairs,
        flow_preds=None,
        flow_gts=None,
        titles=None,
        info=None,
        flow_mag_max=None):
    """Display the given set of image pairs, optionally with flows and titles.
    Args:
        img_pairs: image pairs in [batch_size, 2, H, W, 3] or list([2, H, W, 3]) format.
        flow_preds: optional, predicted flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        flow_gts: optional, groundtruth flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        titles: optional, list of image and flow IDs to display with each image.
        info: optional, stats to display above predicted flow
        flow_mag_max: Max flow to map to 255
    """
    plt = plot_img_pairs_w_flows(img_pairs, None, 0, flow_preds, flow_gts, titles, info, flow_mag_max)
    plt.show()


def archive_img_pairs_w_flows(
        img_pairs,
        dst_file,
        flow_preds=None,
        flow_gts=None,
        titles=None,
        info=None,
        flow_mag_max=None):
    """Plot and save to disk te given set of image pairs, optionally with flows and titles.
    Args:
        img_pairs: image pairs in [batch_size, 2, H, W, 3] or list([2, H, W, 3]) format.
        dst_file: Path where to save resulting image
        flow_preds: optional, predicted flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        flow_gts: optional, groundtruth flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        titles: optional, list of image and flow IDs to display with each image.
        info: optional, stats to display above predicted flow
        flow_mag_max: Max flow to map to 255
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Build plot and save it to disk
    plt = plot_img_pairs_w_flows(img_pairs, None, 0, flow_preds, flow_gts, titles, info, flow_mag_max)
    plt.savefig(dst_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def display_img_pairs_w_flow_pyrs(
        img_pairs,
        flow_pyrs=None,
        num_lvls=0,
        flow_preds=None,
        flow_gts=None,
        titles=None,
        info=None,
        flow_mag_max=None):
    """Display the given set of image pairs, optionally with flows and titles.
    Args:
        img_pairs: image pairs in [batch_size, 2, H, W, 3] or list([2, H, W, 3]) format.
        flow_pyrs: optional, predicted optical flow pyramids [batch_size, H, W, 2] or list([H, W, 2]) format.
        num_lvls: number of levels per pyramid (flow_pyrs must be set)
        flow_preds: optional, predicted flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        flow_gts: optional, groundtruth flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        titles: optional, list of image and flow IDs to display with each image.
        info: optional, stats to display above predicted flow
        flow_mag_max: Max flow to map to 255
    """
    plt = plot_img_pairs_w_flows(img_pairs, flow_pyrs, num_lvls, flow_preds, flow_gts, titles, info, flow_mag_max)
    plt.show()


def archive_img_pairs_w_flow_pyrs(
        img_pairs,
        dst_file,
        flow_pyrs=None,
        num_lvls=0,
        flow_preds=None,
        flow_gts=None,
        titles=None,
        info=None,
        flow_mag_max=None):
    """Plot and save to disk te given set of image pairs, optionally with flows and titles.
    Args:
        img_pairs: image pairs in [batch_size, 2, H, W, 3] or list([2, H, W, 3]) format.
        dst_file: Path where to save resulting image
        flow_pyrs: optional, predicted optical flow pyramids [batch_size, H, W, 2] or list([H, W, 2]) format.
        num_lvls: number of levels per pyramid (flow_pyrs must be set)
        flow_preds: optional, predicted flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        flow_gts: optional, groundtruth flows in [batch_size, H, W, 2] or list([H, W, 2]) format.
        titles: optional, list of image and flow IDs to display with each image.
        info: optional, stats to display above predicted flow
        flow_mag_max: Max flow to map to 255
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Build plot and save it to disk
    plt = plot_img_pairs_w_flows(img_pairs, flow_pyrs, num_lvls, flow_preds, flow_gts, titles, info, flow_mag_max)
    plt.savefig(dst_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
