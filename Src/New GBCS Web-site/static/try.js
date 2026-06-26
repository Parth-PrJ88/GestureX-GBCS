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

function updateWorkflow(phase, progress, message){
    // Current Phase
    document.getElementById("currentPhase").innerText = phase;
    // Progress %
    document.getElementById("workflowPercent").innerText = progress + "%";
    // Progress Bar
    document.getElementById("workflowBar").style.width = progress + "%";
    // Status
    document.getElementById("workflowStatus").innerText = message;

    // Timeline
    const align = document.getElementById("step-align");
    const calibration = document.getElementById("step-calibration");
    const cursor = document.getElementById("step-cursor");
    const line1 = document.getElementById("line-1");
    const line2 = document.getElementById("line-2");

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

    if(progress===0){
        align.classList.add("active");
    }
    else if(progress >= 33 && progress < 66){
        align.classList.add("completed");
        align.querySelector(".circle span").innerHTML = "✓";
        line1.classList.add("completed");
        calibration.classList.add("active");
    }
    else if(progress >=66 && progress < 100){
        align.classList.add("completed");
        align.querySelector(".circle span").innerHTML = "✓";
        line1.classList.add("completed");
        calibration.classList.add("completed");
        calibration.querySelector(".circle span").innerHTML = "✓";
        line2.classList.add("completed");
        cursor.classList.add("active");
    }
    else if(progress >=100){
        align.classList.add("completed");
        align.querySelector(".circle span").innerHTML = "✓";
        line1.classList.add("completed");
        calibration.classList.add("completed");
        calibration.querySelector(".circle span").innerHTML = "✓";
        line2.classList.add("completed");
        cursor.classList.add("completed");
        cursor.querySelector(".circle span").innerHTML = "✓";
    }
}


function fetchWorkflow(){
    fetch("/progress")
    .then(res=>res.json())
    .then(data=>{
        updateWorkflow(
        data.phase,
        data.progress,
        data.message
        );
    })
    .catch(err => {
        updateWorkflow(
            "Error",
            0,
            "Unable to connect to the server."
        );
        console.error(err);
    });
}
// Start only on try.html
if (document.getElementById("workflowBar")) {
    fetchWorkflow();
    setInterval(fetchWorkflow, 500);
}


async function launch(route) {
    try {
        updateWorkflow(
            "Starting...",
            5,
            "Initializing GestureX Model..."
        );

        const response = await fetch(route);
        const data = await response.json();

        console.log("Workflow started:", data);
    }
    catch(err){
        updateWorkflow(
            "Error",
            0,
            "Unable to start workflow."
        );
        console.error(err);
    }
}


async function stopWorkflow() {
    try {
        updateWorkflow(
            "Stopping...",
            0,
            "Stopping current workflow..."
        );

        const response = await fetch("/stop");
        const data = await response.json();
        console.log("Stop request sent:", data);
    } 
    catch (err) {
        console.error(err);
        updateWorkflow(
            "Error",
            0,
            "Unable to stop workflow."
        );
    }
}