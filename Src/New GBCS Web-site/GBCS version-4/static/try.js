/* ================= Floating Background Icons Animation ================= */

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


/* ================= Workflow Dashboard State Manager ================= */

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

// // Status Animation
// let statusAnimation = null;
// let baseStatusText = "";
// let dots = 0;

// function animateStatus(message){
//     // Restart animation only if message changes
//     if(baseStatusText === message)
//         return;
//     baseStatusText = message;
//     clearInterval(statusAnimation);
//     dots = 0;
//     statusAnimation = setInterval(()=>{
//         dots = (dots + 1) % 4;
//         document.getElementById("workflowStatus").innerText =
//             baseStatusText + ".".repeat(dots);
//     },400);
// }

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
    // animateStatus(message);
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


/* ================= Launch, Stop & Reste Workflow ================= */

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
    if(!confirm("Reset Entire Workflow?"))
        return;

    fetch("/reset")
    .then(res=>res.json())
    .then(data=>{
        console.log(data);
        fetchWorkflow();
    });
});


/* ================== Profile Management =================== */

/* ----------------- Load(Fetch) Profiles ------------------ */

async function fetchProfiles(){

    const response = await fetch("/profiles");
    const profiles = await response.json();

    const profileList = document.getElementById("profileList");

    profileList.innerHTML = "";

    if (profiles.length === 0) {
        profileList.innerHTML = `
            <div class="empty-profile">
                No Saved Profiles
            </div>
        `;
        return;
    }

    profiles.forEach(profile => {
        profileList.innerHTML += `
            <div class="profile-card">
                <div class="profile-info">
                    <h3>${profile.name}</h3>
                    <p>${profile.timestamp || "No Timestamp"}</p>
                </div>
                <div class="profile-actions">
                    <button class="load-profile-btn" onclick="loadProfile('${profile.filename}')">
                        <!--📂-->🗃️ Load
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
    await fetchProfiles();
    await fetchActiveProfile();
}

/* --------------------- Save Profile --------------------- */

async function saveProfile(){
    const input = document.getElementById("profileName");
    const profileName = input.value.trim();

    if(profileName === ""){
        alert("Enter Profile Name");
        return;
    }

    const response = await fetch("/profile/save",{
        method: "POST",
        headers: {"Content-Type":"application/json"}, body:JSON.stringify({
            profile_name:profileName
        })
    });

    const result = await response.json();

    alert(result.message);
    input.value="";
    await refreshProfileDashboard();
}

/* --------------------- Load Profile --------------------- */

async function loadProfile(filename){

    // Check complition of Alignmnet Phase before loading a profile
    const currentPhase = document.getElementById("currentPhase").innerText;
    if(currentPhase === "IDLE"){
        alert("Please complete Alignment before loading a Profile.");
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
        // Refresh list & Fetch Active Profile
        await refreshProfileDashboard();
        await fetchWorkflow();
    }else{
        alert(data.message);
    }
}

/* -------------------- Delete Profile -------------------- */

async function deleteProfile(filename){
    const confirmDelete = confirm(
        `Delete "${filename}" profile?\n\nThis action cannot be undone.`
    );

    if(!confirmDelete) return;

    const response = await fetch("/profile/delete",{
        method:"POST",
        headers: {"Content-Type":"application/json"}, body:JSON.stringify({
            filename:filename
        })
    });

    const result = await response.json();
    alert(result.message);
    await refreshProfileDashboard();
}

// Refresh Profile Dashboard
refreshProfileDashboard();
