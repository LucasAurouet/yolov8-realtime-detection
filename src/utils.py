import cv2
import torch
import matplotlib.pyplot as plt

def draw_output(img, pred, t, config, transform=False):
    # Draws an image with the predicted bounding boxes
    # Input : image (torch tensor), predictions , model config
    
    # Extract config 
    S = config['S']
    B = config['B']
    C = config['C']
    
    # Convert the image tensor to numpy array for cv2 and matplotlib
    img = img.permute(1, 2, 0).numpy().copy()
    img_width, img_height = img.shape[:2]

    # x, y, w, h, confidence predictions
    # (S, S, B * 5 + C) -> (S, S, B, 5)
    pred_boxes = pred[..., 0:B * 5].reshape(S, S, B, 5)
    # One-hot predicted classes
    # (S, S, B * 5 + C) -> (S, S, B, C)
    pred_classes = pred[..., B * 5:B * 5 + C]
    
    # Relative corrdinates
    pred_coord = pred_boxes[..., :4] # (N, S, S, B, 4)
    # Confidence 
    pred_conf = pred_boxes[..., 4] # (N, S, S, B)
    
    # Apply the output transformations 
    if transform:
        pred_xy = pred_coord[..., :2]
        pred_wh = torch.exp(pred_coord[..., 2:4])
        pred_coord = torch.cat([pred_xy, pred_wh], dim=-1)
    
    for i in range(S):
        for j in range(S):
            for b in range(B):
                if pred_conf[i,j,b] > t:
                    x, y, w, h = pred_coord[i,j,b]
                    # Transform relative (cell) coordinates to absolute (image) coordinates
                    x_center = (j + x) / S  
                    y_center = (i + y) / S
                    # Transform the absolute [0, 1] coordinates to pixel values
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    # Box dimensions (pixels)
                    w_px = w * img_width
                    h_px = h * img_height
                    # Box corners
                    x1 = int(x_center_px - w_px / 2)
                    y1 = int(y_center_px - h_px / 2)
                    x2 = int(x_center_px + w_px / 2)
                    y2 = int(y_center_px + h_px / 2)
                    
                    # Class and confidence for the current cell
                    class_id = pred_classes[i,j].argmax().item()
                    score = pred_conf[i,j,b].item()
                    # Draw the box
                    color = (0,255,0)
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
                    cv2.putText(img, f"{class_id}:{score:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            # Draws the cells 
            cell_x1 = int(i * (img_width / S))
            cell_x2 = int((i + 1) * (img_width / S))
            cell_y1 = int(j * (img_height / S))
            cell_y2 = int((j + 1) * (img_height / S))
            cv2.rectangle(img, (cell_x1, cell_y1), (cell_x2, cell_y2), (0,0,255))
    
    # Show the image 
    plt.imshow(img)
    plt.axis("off")
    plt.show()
 
def compute_iou(pred_boxes, true_boxes):
    # Compute the IOU of each boxes in a cell

    # Convert x, y, w, h into x1y1, x2y2 (upper left, lower right)
    pred_x1y1 = pred_boxes[..., :2] - pred_boxes[..., 2:] / 2
    pred_x2y2 = pred_boxes[..., :2] + pred_boxes[..., 2:] / 2
    true_x1y1 = true_boxes[..., :2] - true_boxes[..., 2:] / 2
    true_x2y2 = true_boxes[..., :2] + true_boxes[..., 2:] / 2
    
    # Intersection
    inter_x1y1 = torch.max(pred_x1y1, true_x1y1)
    inter_x2y2 = torch.min(pred_x2y2, true_x2y2)
    inter_wh = (inter_x2y2 - inter_x1y1)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    
    # Union
    pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
    true_area = true_boxes[..., 2] * true_boxes[..., 3]
    union_area = pred_area + true_area - inter_area

    ious = torch.zeros_like(union_area)
    ious = inter_area / union_area

    return ious
   
def ind_ij(self, pred_boxes, true_boxes, obj_mask):
    # Indicator vector to detect the boxes responsible for the detection
    
    # IOU
    ious = self.compute_iou(pred_boxes, true_boxes) 
    
    # Create the indicator vector 
    ind = torch.zeros_like(ious)
    
    # Find the boxes responsible for the object
    b_max = torch.argmax(ious, dim=-1, keepdim=True) 
    ind.scatter_(-1, b_max, 1.0)
    
    # if no object, ind_ij = 0
    ind = ind * obj_mask.unsqueeze(-1)
    
    return ind, ious