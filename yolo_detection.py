import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

# %matplotlib inline


def load_yolo_model(model_path):
    """Load the course YOLO model with both legacy and Keras 3 formats."""
    saved_model_path = os.path.join(model_path, "saved_model.pb")
    if (
        os.path.isdir(model_path)
        and os.path.exists(saved_model_path)
        and hasattr(tf.keras.layers, "TFSMLayer")
    ):
        inputs = tf.keras.Input(shape=(608, 608, 3), name="input_1")
        saved_model_layer = tf.keras.layers.TFSMLayer(
            model_path,
            call_endpoint="serving_default"
        )
        outputs = saved_model_layer(inputs)
        if isinstance(outputs, dict):
            outputs = outputs["conv2d_22"]
        return tf.keras.Model(inputs, outputs)

    return load_model(model_path, compile=False)


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold"
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


# UNIT TEST
tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.5)

print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!")
# END UNIT TEST


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou


# Test cases for IoU
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value."

# Test case 2: boxes do not intersect
box1 = (1, 2, 3, 4)
box2 = (5, 6, 7, 8)
print("iou for non-intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

# Test case 3: boxes intersect at vertices only
box1 = (1, 1, 2, 2)
box2 = (2, 2, 3, 3)
print("iou for boxes that only touch at vertices = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

# Test case 4: boxes intersect at edge only
box1 = (1, 1, 3, 3)
box2 = (2, 3, 3, 4)
print("iou for boxes that only touch at edges = " + str(iou(box1, box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"

print("\033[92m All tests passed!")
# END UNIT TEST


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)

    nms_indices = []
    classes_labels = tf.unique(classes)[0]  # Get unique classes
    
    for label in classes_labels:
        filtering_mask = classes == label
    
        # Get boxes for this class
        boxes_label = tf.boolean_mask(boxes, filtering_mask)
        
        # Get scores for this class
        scores_label = tf.boolean_mask(scores, filtering_mask)
        
        if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process
            
            # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
            nms_indices_label = tf.image.non_max_suppression(
                boxes_label,
                scores_label,
                max_output_size=max_boxes,
                iou_threshold=iou_threshold
            )

            # Get original indices of the selected boxes
            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)
            
            # Append the resulting boxes into the partial result
            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

    # Flatten the list of indices and concatenate
    nms_indices = tf.concat(nms_indices, axis=0)
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    # Sort by scores and return the top max_boxes
    sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
    scores = tf.gather(scores, sort_order[0:max_boxes])
    boxes = tf.gather(boxes, sort_order[0:max_boxes])
    classes = tf.gather(classes, sort_order[0:max_boxes])

    return scores, boxes, classes


# Test cases for NMS
scores = np.array([0.855, 0.828])
boxes = np.array([[0.45, 0.2, 1.01, 2.6], [0.42, 0.15, 1.7, 1.01]])
classes = np.array([0, 1])

print(f"iou:    \t{iou(boxes[0], boxes[1])}")

scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold=0.9)

assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
assert np.allclose(boxes2.numpy(), [[0.45, 0.2, 1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes"
assert np.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes"

scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold=0.1)

assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores"
assert np.allclose(boxes2.numpy(), [[0.45, 0.2, 1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes"
assert np.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes"

classes = np.array([0, 0])

# If both boxes belong to the same class, one box is suppressed if iou is under the iou_threshold
scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold=0.15)

assert np.allclose(scores2.numpy(), [0.855]), f"Wrong value on scores"
assert np.allclose(boxes2.numpy(), [[0.45, 0.2, 1.01, 2.6]]), f"Wrong value on boxes"
assert np.array_equal(classes2.numpy(), [0]), f"Wrong value on classes"

print(f"scores:  \t{scores2.numpy()}")
print(f"boxes:  \t{boxes2.numpy()}")     
print(f"classes:\t{classes2.numpy()}")

# If both boxes belong to the same class, one box is suppressed if iou is under the iou_threshold
scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, [0, 0], iou_threshold=0.9)

assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores"
assert np.allclose(boxes2.numpy(), [[0.45, 0.2, 1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes"
assert np.array_equal(classes2.numpy(), [0, 0]), f"Wrong value on classes"

try:
    from unit_tests import test_yolo_non_max_suppression
except ImportError:
    print("Skipping external unit_tests.py checks; file not found.")
else:
    test_yolo_non_max_suppression(yolo_non_max_suppression)
# END UNIT TEST


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores,
    box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Use the function `yolo_filter_boxes` to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(
        boxes,
        box_confidence,
        box_class_probs,
        threshold=score_threshold
    )
    
    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)
    
    # Use the function `yolo_non_max_suppression` to perform Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(
        scores,
        boxes,
        classes,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold
    )
    
    return scores, boxes, classes


# Test yolo_eval
tf.random.set_seed(10)
yolo_outputs = (
    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
    tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
)
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"

print("\033[92m All tests passed!")
# END UNIT TEST


# Load class names and anchors
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608)  # Same as yolo_model input layer size

# Load YOLO model
yolo_model = load_yolo_model("model_data")

# Display model summary
yolo_model.summary()


def predict(image_file):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(
        yolo_outputs,
        [image.size[1], image.size[0]],
        10,
        0.3,
        0.5
    )

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    
    # Generate colors for drawing bounding boxes
    colors = get_colors_for_classes(len(class_names))
    
    # Draw bounding boxes on the image file
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    
    # Save the predicted bounding box on the image
    os.makedirs("out", exist_ok=True)
    image.save(os.path.join("out", image_file), quality=100)
    
    # ===== FIXED: Display the image in a window =====
    # This will open a matplotlib window showing the image with bounding boxes
    plt.figure(figsize=(12, 8))  # Create a figure window
    plt.imshow(image)  # Display the image with bounding boxes already drawn
    plt.axis('off')  # Hide the axes
    plt.title(f"YOLO Detection Results - {len(out_boxes)} objects found")
    plt.show()  # Show the window
    
    # Also print coordinates in a cleaner format
    print("\nDetection Details:")
    for i in range(len(out_scores)):
        box = out_boxes[i].numpy().astype(int)
        print(f"   {class_names[out_classes[i]]}: {out_scores[i]:.2f} at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    
    # Return the results
    return out_scores, out_boxes, out_classes


# Run prediction on test image
print("\n" + "="*50)
print("Running YOLO Object Detection")
print("="*50 + "\n")

out_scores, out_boxes, out_classes = predict("test.jpg")

print("\nDetection completed! Image saved to 'out/test.jpg'")