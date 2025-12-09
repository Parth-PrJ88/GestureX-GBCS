# GestureX-GBCS
Gesture Based Control System (Prototype)

This document outlines the planned steps and features for the GBCS (Gesture-Based Control System) prototype. It includes camera preview, calibration, cursor control, and optional interaction modules.

---

## 🚀 1. Camera Preview (User Alignment Window)

A small camera preview window (230×240) helps the user see their face and position themselves correctly.

### Includes:
- **Alignment Reference**  
  A visual guide showing where the user should position their head.
- **Detection Inaccuracy Warning**  
  Displays a message if the user is too far or the lighting is too poor for accurate face tracking.
- **Transparent Human Outline (Overlay)**  
  A translucent silhouette (head + shoulders) to help users align themselves properly.
- **Dynamic Border Feedback**  
  The outer line of the translucent outline changes color:  
  - **Green** → correct head position & lighting  
  - **Red** → incorrect positioning or poor lighting

### During Calibration:
- The camera preview is **removed** to keep the screen clean.  
- If the user’s head becomes unrecognizable or the background becomes too dark:  
  - The preview **reappears in red**  
  - Calibration is **paused**  
  - Process **automatically resumes** when conditions are corrected

---

## 🎯 2. Calibration System

Calibration collects gaze samples from multiple screen regions.  
**More samples per point → better accuracy.**

### Calibration Details:
- **Point Grid Structure**  
  - Prototype: **3×3 grid**  
  - Future: **4×4 grid** plus **corner points** and **drag-transition animations**
- **Full-Screen Auto Mapping**  
  Uses `pyautogui` to scale correctly to any screen resolution.
- **Bright White Calibration Screen**  
  Ensures the user's face is illuminated even in dark environments.
- **Multi-Stage Calibration**  
  - First calibration on the **normal screen**  
  - Remaining four calibration passes on a **black screen** for accuracy
- **Dynamic Calibration Point Behavior**  
  - Points start **large**, then **shrink** as samples are captured  
  - Movement between points is a **smooth drag animation**, not a sudden jump  
    → reduces user confusion and increases tracking stability
- **Live Condition Monitoring**  
  - If face visibility or background lighting becomes poor → calibration pauses  
  - A red camera preview appears to instruct the user to adjust  
  - Calibration resumes only after conditions return to normal

---

## 🖱️ 3. Cursor Movement (Web-Based Prototype)

Cursor control takes place on a **live interactive webpage**.

### Features:
- **Dual Control Mode**  
  User can operate the webpage using both:
  - Standard mouse
  - Eye-tracking cursor
- **Stabilized Gaze Cursor**  
  The default pointer is replaced with a **large circular marker**, similar to professional eye-trackers, to reduce jitter.
- **Heat Map Tracking**  
  All cursor movement is:
  - Recorded as a **screen recording**
  - Displayed with **heat-map overlays** showing gaze concentration and movement paths

---

## 🔧 4. Interaction Implementation (Optional for Prototype)

If time permits before the deadline, interaction features will be added.

### Planned Tools:
- **PyAutoGUI** – For interaction actions (clicks, dwell, gestures)
- **JSON** – For storing preferences and interaction rules

These will enable basic interaction modes such as:
- Blink-triggered clicks  
- Dwell-based selections  
- Gesture-driven actions  

---

## 📌 Notes

This README describes the prototype workflow and planned enhancements for GBCS. More detailed technical documentation, architecture diagrams, and module descriptions will be added as the project evolves.

---
