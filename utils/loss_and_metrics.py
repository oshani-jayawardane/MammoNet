"""
Authors: Oshani Jayawardane, Buddika Weerasinghe

This file contains the functions for loss and metrics for the mammogram images. The following functions are included:

Metrics:
    - jaccard_coef(y_true, y_pred): Jaccard Coefficient (IoU) (for binary segmentation)
    - jaccard_coef_multiclass(y_true, y_pred): Jaccard Coefficient (IoU) for multi-class segmentation
    
    - dice_coef(y_true, y_pred): Dice Coefficient (for binary segmentation)
    - dice_coef_multiclass(y_true, y_pred): Dice Coefficient for multi-class segmentation
    
    - specificity(y_true, y_pred): Specificity (for binary segmentation)
    - specificity_multiclass(y_true, y_pred): Specificity for multi-class segmentation
    
    - precision(y_true, y_pred): Precision (for binary segmentation)
    - precision_multiclass(y_true, y_pred): Precision for multi-class segmentation
    
    - recall(y_true, y_pred): Recall (for binary segmentation)
    - recall_multiclass(y_true, y_pred): Recall for multi-class segmentation
    
    - f1score(y_true, y_pred): F1 Score (for binary segmentation)
    - f1score_multiclass(y_true, y_pred): F1 Score for multi-class segmentation

Loss Functions:        
    - jaccard_loss(y_true, y_pred): Jaccard Loss (for binary segmentation)
    - jaccard_loss_multiclass(y_true, y_pred): Jaccard Loss for multi-class segmentation
      
    - dice_loss(y_true, y_pred): Dice Loss (for binary segmentation)     
    - dice_loss_multiclass(y_true, y_pred): Dice Loss for multi-class segmentation
    
    - combined_jaccard_dice_loss(alpha=0.5): Combined Jaccard-Dice Loss (for binary segmentation)
    - combined_jaccard_dice_loss_multiclass(alpha=0.5): Combined Jaccard-Dice Loss for multi-class segmentation
    
    - focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25): Focal Loss (for binary segmentation)
    - focal_loss_multiclass(y_true, y_pred, gamma=2.0, alpha=0.25): Focal Loss for multi-class segmentation

"""


import tensorflow as tf

#########################
# losses 
#########################

# Jaccard Coefficient (IoU)
def jaccard_coef(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(T * P)
    union = tf.reduce_sum(T) + tf.reduce_sum(P) - intersection
    # jc = (intersection + 1.0) / (union + 1.0)
    jc = intersection / (union + tf.keras.backend.epsilon())
    return jc

def jaccard_coef_multiclass(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    jc_sum = tf.constant(0, dtype=tf.float32)
    for class_id in range(num_classes):
        y_true_c = y_true[..., class_id]
        y_pred_c = y_pred[..., class_id]
        jc_sum += jaccard_coef(y_true_c, y_pred_c)
    return jc_sum / tf.cast(num_classes, tf.float32)


# Jaccard Loss
def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

def jaccard_loss_multiclass(y_true, y_pred):
    return 1 - jaccard_coef_multiclass(y_true, y_pred)


# Dice Coefficient
def dice_coef(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(T * P)
    summation = tf.reduce_sum(T) + tf.reduce_sum(P)
    # dice = (2.0 * intersection + 1.0) / (summation + 1.0)
    dice = (2.0 * intersection) / (summation + tf.keras.backend.epsilon())
    return dice

def dice_coef_multiclass(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    dice_sum = tf.constant(0, dtype=tf.float32)
    for class_id in range(num_classes):
        y_true_c = y_true[..., class_id]
        y_pred_c = y_pred[..., class_id]
        dice_sum += dice_coef(y_true_c, y_pred_c)
    return dice_sum / tf.cast(num_classes, tf.float32)


# Dice Loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_loss_multiclass(y_true, y_pred):
    return 1 - dice_coef_multiclass(y_true, y_pred)


# Combined Jaccard-Dice Loss
def combined_jaccard_dice_loss(alpha=0.5):
    def loss(y_true, y_pred):
        return alpha * jaccard_loss(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)
    return loss

def combined_jaccard_dice_loss_multiclass(alpha=0.5):
    def loss(y_true, y_pred):
        return alpha * jaccard_loss_multiclass(y_true, y_pred) + (1 - alpha) * dice_loss_multiclass(y_true, y_pred)
    return loss


# Focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = float(y_true)
        y_pred = float(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        focal_loss = -alpha * y_true * tf.math.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred) - (1.0 - alpha) * (1.0 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(focal_loss)
    return loss

def focal_loss_multiclass(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        num_classes = tf.shape(y_true)[-1]
        focal_loss_sum = tf.constant(0, dtype=tf.float32)
        for class_id in range(num_classes):
            y_true_c = y_true[..., class_id]
            y_pred_c = y_pred[..., class_id]
            focal_loss_sum += focal_loss(y_true_c, y_pred_c, gamma, alpha)
        return focal_loss_sum / tf.cast(num_classes, tf.float32)
    return loss

# in focal loss there's no need to flatten, because operations are anyway performed element-wise

#########################
# metrics 
#########################

# Specificity
def specificity(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)

    true_negatives = tf.reduce_sum((1 - T) * (1 - P))
    false_positives = tf.reduce_sum((1 - T) * P)

    specificity = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())
    
    return specificity

def specificity_multiclass(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    specificity_sum = tf.constant(0, dtype=tf.float32)
    for class_id in range(num_classes):
        y_true_c = y_true[..., class_id]
        y_pred_c = y_pred[..., class_id]
        specificity_sum += specificity(y_true_c, y_pred_c)
    return specificity_sum / tf.cast(num_classes, tf.float32)


# Precision
def precision(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)

    true_positives = tf.reduce_sum(T * P)
    false_positives = tf.reduce_sum((1 - T) * P)

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    
    return precision

def precision_multiclass(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    precision_sum = tf.constant(0, dtype=tf.float32)
    for class_id in range(num_classes):
        y_true_c = y_true[..., class_id]
        y_pred_c = y_pred[..., class_id]
        precision_sum += precision(y_true_c, y_pred_c)
    return precision_sum / tf.cast(num_classes, tf.float32)


# Recall
def recall(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)

    true_positives = tf.reduce_sum(T * P)
    false_negatives = tf.reduce_sum(T * (1 - P))

    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    
    return recall

def recall_multiclass(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    recall_sum = tf.constant(0, dtype=tf.float32)
    for class_id in range(num_classes):
        y_true_c = y_true[..., class_id]
        y_pred_c = y_pred[..., class_id]
        recall_sum += recall(y_true_c, y_pred_c)
    return recall_sum / tf.cast(num_classes, tf.float32)


# F1 Score
def f1score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1score = 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())
    return f1score

def f1score_multiclass(y_true, y_pred):
    precision_value = precision_multiclass(y_true, y_pred)
    recall_value = recall_multiclass(y_true, y_pred)
    f1score = 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())
    return f1score