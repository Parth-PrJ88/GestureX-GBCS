/* =============== Floating Background Icons Animation =============== */

const container = document.getElementById("floating-bg");

const iconsList = [
    "fa-hand",
    "fa-eye",
    "fa-microchip",
    "fa-code",
    "fa-brain",
    "fa-computer-mouse",
    "fa-camera",
    "fa-cube"
];

const ICON_COUNT = 15;  // number of floating icons

for (let i = 0; i < ICON_COUNT; i++) {
    createIcon(true); // instant start
}

function createIcon(immediate = false) {
    const icon = document.createElement("i");
    const randomIcon = iconsList[Math.floor(Math.random() * iconsList.length)];
    icon.className = `fa-solid ${randomIcon} icon`;

    resetIcon(icon, immediate);
    container.appendChild(icon);

    icon.addEventListener("animationend", () => {
        resetIcon(icon, false); // recycle
    });
}

function resetIcon(icon, immediate) {
    icon.style.left = Math.random() * 100 + "%";
    icon.style.fontSize = (1.6 + Math.random() * 2.4) + "rem";  //Size of icons

    const duration = 30 + Math.random() * 30;
    icon.style.animationDuration = duration + "s";

    // For start icons at random progress
    icon.style.animationDelay = immediate
        ? `-${Math.random() * duration}s`
        : "1.5s";

    // restart animation
    icon.style.animationName = "none";
    icon.offsetHeight; // force reflow
    icon.style.animationName = "floatUp";
}  


/* ================= Workflow Dashboard State Manager ================ */

/* ----------------- DOM Elements ----------------- */

// Workflow CardBtn
const alignBtn = document.getElementById("alignmentBtn");
const calibrationBtn = document.getElementById("calibrationBtn");
const cursorBtn = document.getElementById("cursorBtn");
const resetBtn= document.getElementById("resetBtn");

// PhaseCards
const alignmentCard = document.getElementById("alignmentCard");
const calibrationCard = document.getElementById("calibrationCard");
const cursorCard = document.getElementById("cursorCard");

// TimeLine
const align = document.getElementById("step-align");
const calibration = document.getElementById("step-calibration");
const cursor = document.getElementById("step-cursor");
const line1 = document.getElementById("line-1");
const line2 = document.getElementById("line-2");

/* ----------------- Animation Helper ----------------- */

// PhaseName-TypeWritter Animation

// ProgressPercentages Animation
let displayedProgress = 0;
let progressAnimation = null;

function animateProgress(target){
    clearInterval(progressAnimation);
    progressAnimation = setInterval(()=>{
        if(displayedProgress < target){
            displayedProgress++;
        }
        else if(displayedProgress > target){
            displayedProgress--;
        }
        else{
            clearInterval(window.progressAnimation);
        }
        document.getElementById("workflowPercent").innerText = displayedProgress + "%";
        document.getElementById("workflowBar").style.width = displayedProgress + "%";
    },48);     // Speed(ms)
}


/* ----------------- Dashboard Workflow ----------------- */

// UpdateDashboard
// ==================================================================
function updateDashboard(phase, progress, message){
    // Current Phase
    document.getElementById("currentPhase").innerText = phase;
    // Progress
    animateProgress(progress);
    // Status
    document.getElementById("workflowStatus").innerText = message;
}

// ResetTimeline
// ==================================================================
function resetTimeline(){
    // Remove previous state
    align.classList.remove("active","completed");
    calibration.classList.remove("active","completed");
    cursor.classList.remove("active","completed");
    line1.classList.remove("completed");
    line2.classList.remove("completed");

    // Reset timeline numbers
    align.querySelector(".circle span").innerHTML = "1";
    calibration.querySelector(".circle span").innerHTML = "2";
    cursor.querySelector(".circle span").innerHTML = "3";
}

// Phasecard ResetBtn Design
// ==================================================================
function resetButtons(){
    alignBtn.disabled=false;
    calibrationBtn.disabled=true;
    cursorBtn.disabled=true;

    alignBtn.classList.remove("running");
    calibrationBtn.classList.remove("running");
    cursorBtn.classList.remove("running");

    // Reset text
    alignBtn.innerHTML="▶ Run";
    calibrationBtn.innerHTML="🔒Locked";
    cursorBtn.innerHTML="🔒Locked";
}

// ResetPhaseCards
// ==================================================================
function resetCards(){
    alignmentCard.classList.remove("active-card","completed-card");
    calibrationCard.classList.remove("active-card","completed-card");
    cursorCard.classList.remove("active-card","completed-card");
}

// UpdateWorkflow
// ==================================================================

//IdleState
function setIdleState(){
    alignmentCard.classList.add("active-card");
    align.classList.add("active");
    alignBtn.disabled = false;
    alignBtn.innerHTML = "▶ Run";
}

// AlignmentState
function setAlignmentState(){
    alignBtn.innerHTML="⏳ Running...";
    alignBtn.classList.add("running");
    alignmentCard.classList.add("active-card");
    align.classList.add("active");
}

// CalibrationState
function setCalibrationState(){
    alignmentCard.classList.add("completed-card");
    calibrationCard.classList.add("active-card");

    alignBtn.disabled=true;
    alignBtn.innerHTML="✓ Completed";
    calibrationBtn.disabled=false;
    calibrationBtn.innerHTML="▶ Run";

    align.classList.add("completed");
    align.querySelector(".circle span").innerHTML = "✓";
    line1.classList.add("completed");
    calibration.classList.add("active");
}
function setCalibrationRunningState(){

    alignmentCard.classList.add("completed-card");
    calibrationCard.classList.add("active-card");

    alignBtn.disabled = true;
    alignBtn.innerHTML = "✓ Completed";
    calibrationBtn.disabled = true;
    calibrationBtn.innerHTML = "⏳ Running...";
    calibrationBtn.classList.add("running");

    align.classList.add("completed");
    align.querySelector(".circle span").innerHTML = "✓";
    line1.classList.add("completed");
    calibration.classList.add("active");
}

// CursorState
function setCursorState(){
    alignmentCard.classList.add("completed-card");
    calibrationCard.classList.add("completed-card");
    cursorCard.classList.add("active-card");

    alignBtn.disabled=true;
    alignBtn.innerHTML="✓ Completed";
    calibrationBtn.disabled=true;
    calibrationBtn.innerHTML="✓ Completed";
    cursorBtn.disabled=false;
    cursorBtn.innerHTML="▶ Run";

    align.classList.add("completed");
    align.querySelector(".circle span").innerHTML = "✓";
    line1.classList.add("completed");
    calibration.classList.add("completed");
    calibration.querySelector(".circle span").innerHTML = "✓";
    line2.classList.add("completed");
    cursor.classList.add("active");
}
function setCursorRunningState(){

    alignmentCard.classList.add("completed-card");
    calibrationCard.classList.add("completed-card");
    cursorCard.classList.add("active-card");

    alignBtn.disabled=true;
    alignBtn.innerHTML="✓ Completed";
    calibrationBtn.disabled=true;
    calibrationBtn.innerHTML="✓ Completed";
    cursorBtn.disabled = true;
    cursorBtn.innerHTML = "⏳ Running...";
    cursorBtn.classList.add("running");

    align.classList.add("completed");
    align.querySelector(".circle span").innerHTML = "✓";
    line1.classList.add("completed");
    calibration.classList.add("completed");
    calibration.querySelector(".circle span").innerHTML = "✓";
    line2.classList.add("completed");
    // cursor.classList.add("active");
    cursor.classList.add("completed");
    cursor.querySelector(".circle span").innerHTML = "✓";
}

// CompletedState
function setCompletedState(){
    alignmentCard.classList.add("completed-card");
    calibrationCard.classList.add("completed-card");
    cursorCard.classList.add("completed-card");

    alignBtn.disabled=true;
    alignBtn.innerHTML="✓ Completed";
    calibrationBtn.disabled=true;
    calibrationBtn.innerHTML="✓ Completed";
    cursorBtn.disabled=true;
    cursorBtn.innerHTML="✓ Completed";

    align.classList.add("completed");
    align.querySelector(".circle span").innerHTML = "✓";
    line1.classList.add("completed");
    calibration.classList.add("completed");
    calibration.querySelector(".circle span").innerHTML = "✓";
    line2.classList.add("completed");
    cursor.classList.add("completed");
    cursor.querySelector(".circle span").innerHTML = "✓";
}

// Workflow
// ----------------------------------------------------------------------
function updateWorkflow(data){

    const {
        phase,
        progress,
        message,
        running,
        completed
    } = data;

    updateDashboard(phase, progress, message);
    resetTimeline();
    resetButtons();
    resetCards();

    // State Logic
    if (phase === "Idle") {
        setIdleState();
    }
    else if (phase === "Alignment") {
        if (running) {
            setAlignmentState();
        }
        else {
            setCalibrationState();
        }
    }
    else if (phase === "Calibration") {
        if (running) {
            setCalibrationRunningState();
        }
        else {
            setCursorState();
        }
    }
    else if (phase === "Cursor Control") {
        if (running) {
            setCursorRunningState();
        }
        else if (completed) {
            setCompletedState();
        }
    }
}

// FetchWorkFlow
function fetchWorkflow(){
    fetch("/progress")
    .then(res=>res.json())
    .then(data=>{
        updateWorkflow(data);
    })
    .catch(err => {
        updateWorkflow({
            phase: "Error",
            progress: 0,
            message: "Unable to connect to the server.",
            running: false,
            completed: false
        });
        console.error(err);
    });
}
// Start only on try.html
if (document.getElementById("workflowBar")) {
    fetchWorkflow();
    setInterval(fetchWorkflow, 500);
}


/* ============== Launch, Stop & Reste Workflow ============== */

async function launch(route, button) {
    try {
        button.disabled = true;

        updateWorkflow({
            phase: "Starting",
            progress: 5,
            message: "Initializing GestureX Model...",
            running: true,
            completed: false
        });

        const response = await fetch(route);
        const data = await response.json();

        console.log("Workflow started:", data);
    }
    catch(err){
        updateWorkflow({
            phase: "Error",
            progress: 0,
            message: "Unable to start workflow.",
            running: false,
            completed: false
        });
        console.error(err);
    }
}


async function stopWorkflow() {
    try {
        updateWorkflow({
            phase: "Stopping",
            progress: 0,
            message: "Stopping current workflow...",
            running: false,
            completed: false
        });

        const response = await fetch("/stop");
        const data = await response.json();
        console.log("Stop request sent:", data);
    } 
    catch (err) {
        console.error(err);
        updateWorkflow({
            phase: "Error",
            progress: 0,
            message: "Unable to stop workflow.",
            running: false,
            completed: false
        });
        
    }
}

// Reset Workflow
resetBtn.addEventListener("click",()=>{
    showModal({
        title: "Reset Workflow",
        message: `
            Are you sure you want to reset the
            <br>
            <span class="modal-profile-name danger-name">
                Workflow
            </span>
            <br>
            This will <strong>clear the current session</strong> &
            <strong>reset progress</strong>.
            <br><br>
            Do you want to continue?`,
        icon: "💀",
        // icon: "💀 🔄️ ⛓️‍💥",
        type: "danger",
        confirmText: "Reset",
        cancelText: "Cancel",

        onConfirm: async ()=>{
            const response = await fetch("/reset");
            const data = await response.json();

            console.log(data);
            await fetchWorkflow();
            await refreshProfileDashboard();
            showToast(
                "Workflow Reset",
                "GestureX has been reset successfully.",
                "success"
            );
        }
    });
});


/* ================== Profile Management =================== */

/* ----------------- Friendly Timestamp ----------------- */

function formatTimestamp(timestamp){
    if(!timestamp)
        return "Unknown";

    const date = new Date(timestamp.replace(" ","T"));
    const now = new Date();
    const diff = now - date;
    const oneDay = 24 * 60 * 60 * 1000;
    const days = Math.floor(diff / oneDay);
    const time = date.toLocaleTimeString([],{
        hour:"numeric",
        minute:"2-digit"
    });

    if(days === 0){
        return `Today • ${time}`;
    }
    if(days === 1){
        return `Yesterday • ${time}`;
    }
    if(days < 7){
        return `${days} days ago`;
    }

    return date.toLocaleDateString([],{
        day:"numeric",
        month:"short",
        year:"numeric"
    });
}

/* ----------------- Load(Fetch) Profiles ------------------ */
let currentActiveProfile = "None";

async function fetchProfiles(){

    const response = await fetch("/profiles");
    const profiles = await response.json();

    const profileList = document.getElementById("profileList");

    profileList.innerHTML = "";

    if (profiles.length === 0) {
        profileList.innerHTML = `
            <div class="empty-profile">
                <div class="empty-profile-icon">
                    👁️
                    <!--👁️ 👤 💙-->
                </div>
                <h3>No Saved Profiles</h3>
                <p>
                    Complete & Save Calibration to access Cursor Control instantly anytime.
                </p>
            </div>
        `;
        return;
    }

    const activeProfile = currentActiveProfile;

    profiles.forEach(profile => {
        const isActive = profile.name === activeProfile;
        profileList.innerHTML += `
            <div class="profile-card ${isActive ? "active" : ""}">
                <div class="profile-info">
                    <h3>${profile.name}</h3>
                    <p title="${profile.timestamp}">
                        ${formatTimestamp(profile.timestamp)}
                    </p>
                </div>
                <div class="profile-actions">
                    <button class="load-profile-btn" onclick="loadProfile('${profile.filename}')">
                        📂 Load
                    </button>
                    <button class="delete-profile-btn" onclick="deleteProfile('${profile.filename}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    });
}

/* ----------------- Fetch Active Profile ----------------- */

async function fetchActiveProfile(){
    try{
        const response = await fetch("/active-profile");
        const data = await response.json();

        // Validation if any profile exist
        const profile = data.profile && data.profile.trim() !== "" ? data.profile: "None";
        currentActiveProfile = profile;
        await fetchProfiles();

        const badge = document.getElementById("activeProfileBadge");
        // Remove previous state
        badge.classList.remove(
            "status-none",
            "status-unsaved",
            "status-saved"
        );

        switch(data.status){

            case "saved":
                badge.classList.add("status-saved");
                badge.innerHTML = "🟢 " + data.profile;
                break;

            case "unsaved":
                badge.classList.add("status-unsaved");
                badge.innerHTML = "🟡 Unsaved Calibration";
                break;

            default:
                badge.classList.add("status-none");
                badge.innerHTML = "⚪ No Calibration";
                break;
        }
    }
    catch(err){
        console.error(err);
        const badge = document.getElementById("activeProfileBadge");

        badge.classList.remove(
            "status-unsaved",
            "status-saved"
        );

        badge.classList.add("status-none");
        badge.innerHTML = "⚪ No Calibration";
    }
}

// Refresh Profile Dashboard Helper
async function refreshProfileDashboard(){
    await fetchActiveProfile();
    await fetchProfiles();
}

/* --------------------- Save Profile --------------------- */

async function saveProfile(overwrite = false, savedName = null){
    const input = document.getElementById("profileName");
    const profileName = savedName || input.value.trim();

    if(profileName === ""){
        showToast(
            "Profile Name Required",
            "Please enter a profile name before saving.",
            "warning"
        );
        return;
    }

    const response = await fetch("/profile/save",{
        method: "POST",
        headers: {"Content-Type":"application/json"}, body:JSON.stringify({
            profile_name: profileName,
            overwrite: overwrite
        })
    });

    const result = await response.json();

    if(result.success){
        showToast(
            "Profile Saved",
            `"${profileName}" has been saved successfully.`,
            "success"
        );
        input.value="";
        await refreshProfileDashboard();
    }else if(result.duplicate){
        const currentProfileName = profileName;
        showModal({
            title: "Replace Profile",
            message: `A calibration profile named<br>
                    <span class="modal-profile-name warning-name">"${result.existing_name}"</span>
                    <br>already exists.
                    <br>Replacing it will <strong>overwrite</strong> the previous calibration.
                    <br><br>
                    Do you want to replace it?
                    `,
            icon: "📝",
            // icon: "📝 ⚠️ 🔗 ⛓️‍💥",
            type: "warning",
            confirmText: "Replace",
            cancelText: "Cancel",
            onConfirm: () => {
                saveProfile(true, currentProfileName);
            }
        });
    }else{
        showToast(
            "Save Failed",
            result.message,
            "error"
        );
    }

    input.value="";
    await refreshProfileDashboard();
}

/* --------------------- Load Profile --------------------- */

async function loadProfile(filename){

    // Check complition of Alignmnet Phase before loading a profile
    const currentPhase = document.getElementById("currentPhase").innerText;
    if(currentPhase === "IDLE"){
        showToast(
            "Alignment Required",
            "Please complete Alignment before loading a Profile.",
            "warning"
        );
        return;
    }

    const response = await fetch("/profile/load",{
        method:"POST",
        headers: {"Content-Type":"application/json"}, body:JSON.stringify({
            filename:filename
        })
    });

    const data = await response.json();

    if(data.success){
        showToast(
            "Profile Loaded",
            `Profile: "${data.message}" is ready for Cursor Control.`,
            "success"
        );
        // Refresh list & Fetch Active Profile
        await refreshProfileDashboard();
        await fetchWorkflow();
    }else{
        showToast(
            "Alignment Required",
            data.message,
            "warning"
        );
    }
}

/* -------------------- Delete Profile -------------------- */

async function deleteProfile(filename){
    showModal({
        title: "Delete Profile",
        message: `Are you sure you want to delete profile named
                <br><span class="modal-profile-name danger-name">"${filename}"</span>
                <br><br>This action <strong>cannot be undone</strong>.`,
        icon: "🗑️",
        type:"danger",
        confirmText: "Delete",
        cancelText: "Cancel",
        onConfirm: async ()=>{
            const response = await fetch("/profile/delete",{
                method:"POST",
                headers: {"Content-Type":"application/json"},body:JSON.stringify({
                    filename:filename
                })
            });

            const result = await response.json();
            // alert(result.message);
            if(result.success){
                showToast(
                    "Profile Deleted",
                    `Profile: "${filename}" removed successfully.`,
                    "success"
                );
            }else{
                showToast(
                    "Profile Deletion Failed",
                    result.message,
                    "error"
                );
            }
            await refreshProfileDashboard();
        }
    });
}



/* ==================== Toast Alert Notification =================== */

function showToast(title, message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");

    toast.className = `toast toast-${type}`;

    const icons = {
        success: "✔",
        warning: "⚠",
        error: "✖",
        // info: "💡 🚀 🔷 ⚡ 🤖"
        info: "🚀"
    };

    toast.innerHTML = `
        <span class="toast-close">&times;</span>
        <div class="toast-header">
            <div class="toast-icon ${type}">
                ${icons[type]}
            </div>
            <div>
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        </div>
        <div class="toast-progress"></div>
    `;

    // Append toast to container in order
    container.appendChild(toast);
    
    toast.querySelector(".toast-close").addEventListener("click",()=>{
        toast.classList.add("hide");
        setTimeout(()=>{
            toast.remove();
        },400);
    });

    const DURATION = 3000;
    let remaining = DURATION;
    let startTime = Date.now();
    let timeoutId;

    /* Close Toast */
    function closeToast(){
        if(toast.classList.contains("hide"))
            return;
        clearTimeout(timeoutId);
        toast.classList.add("hide");
        setTimeout(() => {
            toast.remove();
        },400);
    }

    /* Start Timer */
    function startTimer(){
        startTime = Date.now();
        timeoutId = setTimeout(closeToast, remaining);
    }

    /* Initial Start */
    startTimer();
    /* Pause */
    toast.addEventListener("mouseenter",()=>{
        clearTimeout(timeoutId);
        remaining -= Date.now() - startTime;
        remaining = Math.max(remaining,0);
        toast.classList.add("paused");
    });
    /* Resume */
    toast.addEventListener("mouseleave",()=>{
        toast.classList.remove("paused");
        startTimer();
    });

    /* Close Button */
    toast.querySelector(".toast-close").addEventListener("click", closeToast);
}


/* ======================= Confirmation Modal ======================== */

let modalCallback = null;

function showModal({
    title,
    message,
    icon = "⚠",
    type="warning",
    confirmText = "Confirm",
    cancelText = "Cancel",
    onConfirm = null
}){
    modalCallback = onConfirm;

    document.getElementById("modalTitle").innerText = title;
    document.getElementById("modalMessage").innerHTML = message;
    document.getElementById("modalIcon").innerText = icon;

    const modalIcon = document.getElementById("modalIcon");
    const modalTitle = document.getElementById("modalTitle");
    const modalConfirm = document.getElementById("modalConfirm");
    modalIcon.className = "modal-icon";
    modalTitle.className = "";
    modalConfirm.className = "modal-btn";

    switch(type){
        case "danger":
            modalIcon.classList.add("modal-danger");
            modalTitle.classList.add("title-danger");
            modalConfirm.classList.add("confirm-danger");
            break;

        case "success":
            modalIcon.classList.add("modal-success");
            modalTitle.classList.add("title-success");
            modalConfirm.classList.add("confirm-success");
            break;

        case "info":
            modalIcon.classList.add("modal-info");
            modalTitle.classList.add("title-info");
            modalConfirm.classList.add("confirm-info");
            break;

        // Warning Modal
        default:
            modalIcon.classList.add("modal-warning");
            modalTitle.classList.add("title-warning");
            modalConfirm.classList.add("confirm-warning");
            break;
    }

    document.getElementById("modalConfirm").innerText = confirmText;
    document.getElementById("modalCancel").innerText = cancelText;

    document.getElementById("modalOverlay").classList.add("show");
}

function closeModal(){
    document.getElementById("modalOverlay").classList.remove("show");
}

// Dismiss Modal on Overlay(Outside) Click
document.getElementById("modalOverlay").addEventListener("click",(e)=>{
    if(e.target.id==="modalOverlay"){
        closeModal();
    }
});

/* ---------------- Modal Buttons ---------------- */

document.getElementById("modalCancel").addEventListener("click", closeModal);

document.getElementById("modalConfirm").addEventListener("click",()=>{
    closeModal();
    if(modalCallback){
        modalCallback();
    }
});



/* ======================== User Guide Modal ========================= */

const guideOverlay = document.getElementById("guideOverlay");

document.getElementById("guideBtn").onclick = ()=>{
    guideOverlay.classList.add("show");
};
document.getElementById("closeGuideBtn").onclick = ()=>{
    guideOverlay.classList.remove("show");
};

guideOverlay.onclick = (e)=>{
    if(e.target===guideOverlay){
        guideOverlay.classList.remove("show");
    }
};

document.addEventListener("keydown",(e)=>{
    if(e.key==="Escape"){
        guideOverlay.classList.remove("show");
    }
});

/* ----------- Guide Slidebar Contents ----------- */

const overviewHTML = `
    <div class="guide-hero">
        <h2>Welcome to <span class>Gesture</span>X 👋</h2>
        <p> GestureX allows user to control their computer using eye tracking technology. 
            Follow the workflow below to achieve the best tracking accuracy.
        </p>
        <div class="guide-tour-start">
            <button class="guide-tour-btn" onclick="startInteractiveTour()">
                🪧 Start Interactive Tour
            </button>
        </div>
        <div class="tour-startup-option">
            <label>
                <input type="checkbox" id="welcomeTourCheckbox" class="tour-startup-checkbox" checked>
                Show Welcome Tour on startup
            </label>
        </div>
    </div>

    <div class="guide-section">
        <h3>🚀 Quick Start Workflow</h3>
        <div class="guide-workflow">
            <div class="guide-step">
                <p><span>1.</span> Run Alignment</p>
            </div>
            <div class="guide-step">
                <p><span>2.</span> Complete Calibration</p>
            </div>
            <div class="guide-step">
                <p><span>3.</span> Save Profiles</p>
            </div>
            <div class="guide-step">
                <p><span>4.</span> Load Profile</p>
            </div>
            <div class="guide-step">
                <p><span>5.</span> Start Cursor Control</p>
            </div>
        </div>
    </div>

    <div class="guide-section">
        <h3>💡Tips for Best Results</h3>
        <div class="tips-grid">
            <div class="tip-card">Sit 45-60 cm from camera</div>
            <div class="tip-card">Use good lighting</div>
            <div class="tip-card">Keep head steady</div>
            <div class="tip-card">Blink eyes naturally</div>
        </div>
    </div>

    <div class="guide-section">
        <h3>🎥 Quick Start Tutorial</h3>
        <div class="video-card">
            <!-- <div class="video-placeholder">
                ▶ Tutorial Video
            </div> -->
            <div class="video-preview">
                <i class="fa-solid fa-circle-play"></i>
            </div>
            <button class="guide-video-btn" onclick="openTutorial('https://youtube.com')">
                ▶ Watch 2-Min Quick Start
            </button>
        </div>
    </div>
`;

const alignmentHTML = `
    <h2>🎯 Alignment Guide</h2>

    <p>Position face correctly before starting the process.</p>

    <div class="guide-card">
        <h3>📷 Camera Placement</h3>
        <ul>
            <li>Camera should be at eye level.</li>
            <li>Keep your face centered.</li>
            <li>Maintain 45-60 cm distance.</li>
            <li>Ensure your face is fully visible.</li>
        </ul>
    </div>

    <div class="guide-card">
        <h3>💡 Before Clicking Run</h3>
        <ul>
            <li>Good room lighting.</li>
            <li>Minimize head movement.</li>
            <li>Blink naturally.</li>
            <li>Avoid strong backlight.</li>
        </ul>
    </div>

    <div class="guide-card">
        <h3>🎥 Tutorial Video</h3>
        <button class="guide-video-btn">
            ▶ Watch Alignment Tutorial
        </button>
    </div>
`;

const calibrationHTML = `
    <h2>👀 Calibration Guide</h2>
    <p>
    Calibration maps your eye movement to the screen.
    Accurate calibration provides smooth and precise cursor control.
    </p>

    <div class="guide-card">
        <h3>🎯 Before Starting</h3>

        <div class="guide-checklist">
            <div class="guide-check-item">
            ✅ Keep your head steady
            </div>

            <div class="guide-check-item">
            ✅ Blink naturally
            </div>

            <div class="guide-check-item">
            ✅ Focus on every target
            </div>

            <div class="guide-check-item">
            ✅ Stay 45-60 cm from camera
            </div>
        </div>
    </div>

    <div class="guide-card">
        <h3>📍 During Calibration</h3>

        <div class="workflow-list">
            <div class="workflow-item">
                <span>1</span>
                <p>Wait until each target is captured.</p>
            </div>

            <div class="workflow-item">
                <span>2</span>
                <p>Move only your eyes.</p>
            </div>

            <div class="workflow-item">
                <span>3</span>
                <p>Don't anticipate the next point.</p>
            </div>

            <div class="workflow-item">
                <span>4</span>
                <p>Complete all 16 calibration points.</p>
            </div>
        </div>
    </div>

    <div class="guide-card">
        <h3>⚠️ Common Mistakes</h3>
        <div class="warning-list">
            <div>❌ Moving your head</div>
            <div>❌ Poor room lighting</div>
            <div>❌ Looking away too early</div>
            <div>❌ Sitting too close</div>
        </div>
    </div>

    <div class="guide-card">
        <h3>🎥 Video Tutorial</h3>
        <button class="guide-video-btn">
            ▶ Watch Calibration Tutorial
        </button>
    </div>

    <div class="guide-tip">
        💡 <strong>Tip:</strong>
        Save your calibration profile once calibration
        is completed. Next time you can simply load
        the saved profile and skip calibration.
    </div>
`;

const cursorHTML = `
    <h2>🖱️ Cursor Control</h2>
    <p>
    After Calibration is complete, You can now control the mouse cursor using your eye movements.
    </p>

    <div class="guide-card">
        <h3>🖱️ How to Use</h3>
        <div class="guide-checklist">
            <div class="guide-check-item">✅ Move your eyes naturally.</div>
            <div class="guide-check-item">✅ Keep your head steady.</div>
            <div class="guide-check-item">✅ Blink normally.</div>
            <div class="guide-check-item">✅ Stay within the camera frame.</div>
        </div>
    </div>

    <div class="guide-card">
        <h3>⌨️ Keyboard Shortcut</h3>
        <div class="shortcut-card">
            <ul>
                <li>
                    <strong>ESC - </strong>
                    Stops Cursor Control
                </li>
                <li>
                    <strong>Ctrl+Q - </strong>
                    Universal Exit
                </li>
            </ul>
        </div>
    </div>

    <div class="guide-card">
        <h3>🎥 Tutorial</h3>
        <button class="guide-video-btn">
            ▶ Watch Cursor Control Tutorial
        </button>
    </div>

    <div class="guide-tip">
        💡 <strong>Tip:</strong>
        If tracking becomes inaccurate,
        reload accurate saved profile or recalibrate.
    </div>
`;

const profilesHTML = `
    <h2>💾 Profile Management</h2>
    <p>
    Profiles allow to save calibrations and reuse it anytime.
    </p>

    <div class="guide-card">
        <h3>🔄 Workflow</h3>
        <div class="workflow-list">
            <div class="workflow-item">
                <span>1</span>
                <p>Complete Alignment</p>
            </div>

            <div class="workflow-item">
                <span>2</span>
                <p>Finish Calibration</p>
            </div>

            <div class="workflow-item">
                <span>3</span>
                <p>Save Profile</p>
            </div>

            <div class="workflow-item">
                <span>4</span>
                <p>Load Profile anytime to skip calibration.</p>
            </div>
        </div>
    </div>

    <div class="guide-card">
        <h3>🔍 Available Features</h3>
        <div class="guide-checklist">
            <div class="guide-check-item">
                💾 Save Calibration
            </div>

            <div class="guide-check-item">
                📂 Load Existing Profile
            </div>

            <div class="guide-check-item">
                🗑️ Delete Profile
            </div>

            <div class="guide-check-item">
                🟢 View Active Profile
            </div>
        </div>
    </div>

    <div class="guide-tip">
        💡 <strong>Recommended:</strong>
        Save a profile after every successful calibration to avoid repeating the process.
    </div>
`;

const tutorialsHTML = `
    <h2>🎥 Tutorial Library</h2>
    <p>
    Watch short tutorials to master GestureX and get best performance.
    </p>

    <div class="tutorial-grid">
        <div class="tutorial-card">
            <h3>🎯 Alignment</h3>
            <p>Camera positioning and face alignment.</p>
            <button class="guide-video-btn">
                ▶ Watch
            </button>
        </div>

        <div class="tutorial-card">
            <h3>👀 Calibration</h3>
            <p>Complete all calibration points correctly.</p>
            <button class="guide-video-btn">
                ▶ Watch
            </button>
        </div>

        <div class="tutorial-card">
            <h3>🖱️ Cursor Control</h3>
            <p>Move your cursor smoothly using eye tracking.</p>
            <button class="guide-video-btn">
                ▶ Watch
            </button>
        </div>
    </div>
`;

const guidePages = {
    overview: overviewHTML,
    alignment: alignmentHTML,
    calibration: calibrationHTML,
    cursor: cursorHTML,
    profiles: profilesHTML,
    tutorials: tutorialsHTML
};

function loadGuidePage(page){
    const content = document.getElementById("guideContent");

    content.style.opacity = "0";
    content.style.transform = "translateY(12px)";

    setTimeout(()=>{
        content.innerHTML = guidePages[page];
        // Sync dynamic checkboxes
        syncTourCheckboxes();
        content.style.opacity = "1";
        content.style.transform = "translateY(0)";
    },180);
}
loadGuidePage("overview");
document.querySelectorAll(".guide-tab").forEach(tab=>{
    tab.addEventListener("click",()=>{
        document.querySelector(".guide-tab.active")?.classList.remove("active");
        tab.classList.add("active");
        loadGuidePage(tab.dataset.page);
    });
});

// Tutorial URL Function
function openTutorial(url){
    window.open(url,"_blank");
}


/* ======================== Wlecome Tour Modal ======================= */

const tourWelcomeOverlay = document.getElementById("tourWelcomeOverlay");
const startTourBtn = document.getElementById("startTourBtn");
const skipTourBtn = document.getElementById("skipTourBtn");

function showTourWelcome(){
    tourWelcomeOverlay.classList.add("show");
}

// Start Tour on First time
startTourBtn.onclick=()=>{
    tourWelcomeOverlay.classList.remove("show");
    currentTourStep=0;
    setTimeout(()=>{
        showTourStep(0);
    },250);
};

// Start Tour from User-Guidance
window.startInteractiveTour = function () {
    // Close the User Guidance modal
    guideOverlay.classList.remove("show");
    // Start the tour from Step 1
    currentTourStep = 0;
    setTimeout(() => {
        showTourStep(0);
    }, 250);
};

// Skip Tour
skipTourBtn.onclick=()=>{
    tourWelcomeOverlay.classList.remove("show");
};

/* ---------- Show Welcome Tour on startup ---------- */

/* Tour Preference */
const TOUR_STORAGE_KEY = "gesturexShowWelcomeTour";
function getTourPreference(){
    // Default value = true
    return localStorage.getItem(TOUR_STORAGE_KEY) !== "false";
}
function setTourPreference(show){
    console.log("Saving:", show);
    localStorage.setItem(TOUR_STORAGE_KEY, show);
    syncTourCheckboxes();
}
function syncTourCheckboxes(){
    const show = getTourPreference();
    document.querySelectorAll(".tour-startup-checkbox")
        .forEach(cb => {
            cb.checked = show;
        });
}

document.addEventListener("change",(e)=>{
    if(e.target.classList.contains("tour-startup-checkbox")){
        setTourPreference(e.target.checked);
    }
});


/* ==================== Interactive Tour Tooltip ===================== */

const tourSteps=[
    {
        highlight:"#guideBtn",
        anchor:"#guideBtn",
        title: "📘 User Guide",
        description: "Learn GestureX through Guides, Tutorials and Workflow Instructions."
    },
    {
        highlight:"#alignmentCard",
        anchor:"#alignmentCard",
        title: "🎯 Alignment",
        description: "Click Run to begin the Face Alignment process."
    },
    {
        highlight: "#workflowDashboard",
        anchor: "#workflowDashboard",
        title: "✔️ Workflow Dashboard",
        description: "Identify the Current Status of Workflow."
    },
    {
        highlight:"#calibrationCard",
        anchor:"#calibrationCard",
        title: "👀 Calibration",
        description: "Click Run & Complete tracing of all Calibration points carefully."
    },
    {
        highlight: "#profileDashboard",
        anchor: "#profileDashboard",
        title: "💾 Profile Dashboard",
        description: "Save, Load and Manage Calibration Profiles."
    },
    {
        highlight:"#cursorCard",
        anchor:"#cursorCard",
        title: "🖱️ Cursor Control",
        description: "Click Run for Controlling the mouse cursor."
    },
    {
        highlight: "#stopBtn",
        anchor: "#stopBtn",
        title: "🛑 Stop",
        description: "Safely Stops any Running process."
    },
    {
        highlight: "#resetBtn",
        anchor: "#resetBtn",
        title: "🔄 Reset",
        description: "Reset the complete workflow."
    }
];

let currentTourStep = 0;

// Remove Hightlight after finish specific tour
function clearHighlight(){
    document.querySelectorAll(".tour-highlight").forEach(el=>{
        el.classList.remove("tour-highlight");
    });
    document.getElementById("tourFrame").style.display="none";
}

// Positioning Tour Tooltip
function positionTooltip(target){

    const tooltip = document.getElementById("tourTooltip");
    const rect = target.getBoundingClientRect();

    const tooltipWidth = tooltip.offsetWidth;
    const tooltipHeight = tooltip.offsetHeight;
    const gap = 20;

    let left, top;
    let placement = "bottom";

    /* ---------- Try Bottom ---------- */
    left = rect.left + (rect.width - tooltipWidth) / 2;
    top = rect.bottom + gap;
    if(top + tooltipHeight <= window.innerHeight){
        placement = "bottom";
    }

    /* ---------- Try Top ---------- */
    else if(rect.top - tooltipHeight - gap >= 0){
        top = rect.top - tooltipHeight - gap;
        placement = "top";
    }

    /* ---------- Try Right ---------- */
    else if(rect.right + tooltipWidth + gap <= window.innerWidth){
        left = rect.right + gap;
        top = rect.top + (rect.height - tooltipHeight) / 2;
        placement = "right";
    }

    /* ---------- Otherwise Left ---------- */
    else{
        left = rect.left - tooltipWidth - gap;
        top = rect.top + (rect.height - tooltipHeight) / 2;
        placement = "left";
    }

    /* Keep inside viewport */
    left = Math.max(
        gap,
        Math.min(left, window.innerWidth - tooltipWidth - gap)
    );

    top = Math.max(
        gap,
        Math.min(top, window.innerHeight - tooltipHeight - gap)
    );

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";

    tooltip.classList.remove(
        "arrow-top",
        "arrow-bottom",
        "arrow-left",
        "arrow-right"
    );

    const arrowMap = {
        bottom: "arrow-top",
        top: "arrow-bottom",
        left: "arrow-right",
        right: "arrow-left"
    };
    tooltip.classList.add(arrowMap[placement]);
}

// Update tooltip Position
function updateTourPosition(){

    const step = tourSteps[currentTourStep];
    if(!step) return;

    const highlightTarget = document.querySelector(step.highlight);
    const anchorTarget = document.querySelector(step.anchor);

    if(!highlightTarget) return;

    // Update frame
    if(step.highlight !== "#guideBtn"){

        const rect = highlightTarget.getBoundingClientRect();
        const frame = document.getElementById("tourFrame");

        frame.style.left = (rect.left - 8) + "px";
        frame.style.top = (rect.top - 8) + "px";
        frame.style.width = (rect.width + 16) + "px";
        frame.style.height = (rect.height + 16) + "px";
    }
    // Update tooltip
    positionTooltip(anchorTarget);
}
let ticking = false;
window.addEventListener("scroll", () => {
    if (!ticking) {
        requestAnimationFrame(() => {
            updateTourPosition();
            ticking = false;
        });
        ticking = true;
    }
});
window.addEventListener("resize", updateTourPosition);

function showTourStep(index){
    clearHighlight();
    document.getElementById("tourOverlay").classList.add("show");
    const step = tourSteps[index];
    const highlightTarget = document.querySelector(step.highlight);
    const anchorTarget = document.querySelector(step.anchor);

    document.getElementById("tourTitle").innerHTML = step.title;
    document.getElementById("tourDescription").innerHTML = step.description;
    // Progress
    document.getElementById("tourCounter").innerHTML = `STEP ${index+1} OF ${tourSteps.length}`;
    // Progress bar
    document.getElementById("tourProgressBar").style.width = ((index+1)/tourSteps.length)*100+"%";
    // Triger Tour-Tooltip
    document.getElementById("tourTooltip").classList.add("show");

    if(!highlightTarget)
        return;

    const frame = document.getElementById("tourFrame");
    const rect = highlightTarget.getBoundingClientRect();

    frame.style.display = "block";
    // Reset
    frame.classList.remove("pulse");
    // Guide Button
    if(step.highlight === "#guideBtn"){
        frame.classList.add("pulse");
        frame.style.borderRadius = "50%";
        const padding = 8;
        frame.style.left = (rect.left - padding) + "px";
        frame.style.top = (rect.top - padding) + "px";
        frame.style.width = (rect.width + padding*2) + "px";
        frame.style.height = (rect.height + padding*2) + "px";
    }
    else{
        frame.style.borderRadius = "18px";
        const padding = 8;
        frame.style.left = (rect.left - padding) + "px";
        frame.style.top = (rect.top - padding) + "px";
        frame.style.width = (rect.width + padding*2) + "px";
        frame.style.height = (rect.height + padding*2) + "px";
    }

    // Tour Tooltip Position 
    positionTooltip(anchorTarget);

    // Change Arrow place of Tooltip only for GuideBtn
    const tooltip = document.getElementById("tourTooltip");
        // Reset first
    tooltip.style.removeProperty("--arrow-left");
    tooltip.style.removeProperty("--arrow-top");
        // Only for User Guide
    if(step.highlight === "#guideBtn"){
        tooltip.style.setProperty("--arrow-left", "34px");
    }

    highlightTarget.scrollIntoView({
        behavior:"smooth",
        block:"center"
    });
}

function endTour(){
    clearHighlight();
    document.getElementById("tourOverlay").classList.remove("show");
    document.getElementById("tourTooltip").classList.remove("show");
    document.getElementById("tourFrame").style.display="none";
    if(currentTourStep >= tourSteps.length){
        setTourPreference(false);
    }
}

/* ---------- Tour Navigation ---------- */
const tourNext = document.getElementById("tourNext");
const tourPrev = document.getElementById("tourPrev");
const tourSkip = document.getElementById("tourSkip");

tourNext.onclick=()=>{
    currentTourStep++;
    if(currentTourStep>=tourSteps.length){
        endTour();
        return;
    }
    showTourStep(currentTourStep);
};
tourPrev.onclick=()=>{
    if(currentTourStep===0)
        return;
    currentTourStep--;
    showTourStep(currentTourStep);
};
tourSkip.onclick=endTour;


/* =========================== Auto-Start ============================ */
/* ------------------------------------------------------------------- */

// Refresh Profile Dashboard
refreshProfileDashboard();
// Welcome Alert
showToast(
    "Gesture<span id='char-X'>X</span>",
    "Welcome Sir! GestureX is Ready to Use.",
    "info"
);
// Tour
window.addEventListener("load",()=>{
    // Check Auto-Tour-Startup on refresh 
    syncTourCheckboxes();
    if(getTourPreference()){
        setTimeout(()=>{
            showTourWelcome();
        },700);
    }
});

