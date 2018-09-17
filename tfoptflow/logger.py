"""
logger.py

Tensor ops-free logger to Tensorboard.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    Written by Michael Gygli
    License: Copyleft

To look at later:
    - Display Image1 and Image2 as an animated GIF rather than side-by-side? Demonstrated below:
    https://github.com/djl11/PWC_Net_TensorFlow/blob/e321b0ee825416a84258559fc90e13cf722e4608/custom_ops/native.py#L216-L242

    - Add error between predicted flow and gt? Demonstrated below:
    https://github.com/djl11/PWC_Net_TensorFlow/blob/e321b0ee825416a84258559fc90e13cf722e4608/custom_ops/native.py#L198-L214
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

from visualize import plot_img_pairs_w_flows


class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, tag=None, graph=None):
        """Creates a summary writer logging to log_dir.
        Args:
            log_dir: Tensorboard logging directory
            tag: Suggested logger types: ('train', 'val', 'test')
            graph: Optional, TF graph
        """
        if graph is None:
            graph = tf.get_default_graph()
        if tag is not None:
            log_dir = log_dir + tag
        self.writer = tf.summary.FileWriter(log_dir, graph=graph)
        self._tag = tag

    @property
    def tag(self):
        if self._tag is None:
            return self._tag
        else:
            return ""

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Args:
            tag: name of the scalar
            value: scalar value to log
            step: training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step, IDs=None):
        """Logs a list of images.
        Args:
            tag: format for the name of the summary (will format ID accordingly)
            images: list of images
            step: training iteration
            IDs: list of IDs
        """
        if images is None:
            return

        im_summaries = []
        for n in range(len(images)):
            # Write the image to a string
            faux_file = BytesIO()  # StringIO()
            if len(images[n].shape) == 3 and images[n].shape[2] == 1:
                image = np.squeeze(images[n], axis=-1)  # (H, W, 1) -> (H, W)
                cmap = 'gray'
            else:
                image = images[n]
                cmap = None
            plt.imsave(faux_file, image, cmap=cmap, format='png')  # (?, H, W, ?)
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=faux_file.getvalue(), height=image.shape[0],
                                       width=image.shape[1])
            # Create a Summary value
            img_tag = tag.format(IDs[n]) if IDs is not None else tag.format(n)
            im_summaries.append(tf.Summary.Value(tag=img_tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class OptFlowTBLogger(TBLogger):
    """Logging of optical flows and pyramids in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, tag=None, graph=None):
        """Creates a summary writer logging to log_dir. See base class inplementation for details.
        """
        super().__init__(log_dir, tag, graph)

    def log_imgs_w_flows(self, tag, img_pairs, flow_pyrs, num_lvls, flow_preds, flow_gts, step, IDs=None, info=None):
        """Logs a list of optical flows.
        Args:
            tag: format for the name of the summary (will format ID accordingly)
            img_pairs: list of image pairs in [batch_size, 2, H, W, 3]
            flow_pyrs: predicted optical flow pyramids [batch_size, H, W, 2] or list([H, W, 2]) format.
            num_lvls: number of levels per pyramid (flow_pyrs must be set)
            flow_preds: list of predicted flows in [batch_size, H, W, 2]
            flow_gts: optional, list of groundtruth flows in [batch_size, H, W, 2]
            step: training iteration
            IDs: optional, list of IDs
        """
        assert(len(img_pairs) == len(flow_preds))
        im_summaries = []
        for n in range(len(img_pairs)):
            # Combine image pair, predicted flow, flow pyramid, and groundtruth flow in a single plot
            img_pair = np.expand_dims(img_pairs[n], axis=0)
            flow_pyr = np.expand_dims(flow_pyrs[n], axis=0) if flow_pyrs is not None else None
            flow_pred = np.expand_dims(flow_preds[n], axis=0) if flow_preds is not None else None
            gt_flow = np.expand_dims(flow_gts[n], axis=0) if flow_gts is not None else None

            plt = plot_img_pairs_w_flows(img_pair, flow_pyr, num_lvls, flow_pred, gt_flow, None, info)

            # Write the image to a string
            faux_file = BytesIO()  # StringIO()
            plt.savefig(faux_file, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=faux_file.getvalue())

            # Create a Summary value
            img_tag = tag.format(IDs[n]) if IDs is not None else tag.format(n)
            im_summaries.append(tf.Summary.Value(tag=img_tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
