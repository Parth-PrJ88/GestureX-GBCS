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

/* ================= Toggle Menu ================= */

const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');
const navBtns = document.querySelector('.nav-btns');

navToggle.addEventListener('click', () => {
  navToggle.classList.toggle('active');
  navLinks.classList.toggle('active');
  navBtns.classList.toggle('active');
});

/* ================= Hide and show navigation on scroll ================= */

let lastScroll = 0;
const nav = document.querySelector(".navbar");

window.addEventListener("scroll", () => {
  let current = window.scrollY;

  if (current > lastScroll && current > 100) {
    nav.style.transform = "translateY(-150%)";
    nav.style.transition = "transform 0.5s ease";
  } else {
    nav.style.transform = "translateY(0)";
    nav.style.transition = "transform 0.5s ease"; 
    
  }
  lastScroll = current;
});

/* ================= Active Navbar on Scroll ================= */

// const sections = document.querySelectorAll(".main-section");
const sections = document.querySelectorAll("#home, #project, #projectmembers");
const navLinksAll = document.querySelectorAll(".nav-links a");

window.addEventListener("scroll", () => {
  let currentSection = "";

  sections.forEach((section) => {
    const sectionTop = section.offsetTop - 150; // adjust for navbar height
    const sectionHeight = section.clientHeight;

    if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
      currentSection = section.getAttribute("id");
    }
  });

  navLinksAll.forEach((link) => {
    link.classList.remove("active");

    if (link.getAttribute("href") === "#" + currentSection) {
      link.classList.add("active");
    }
  });
});

/* ================= Active on Click ================= */

navLinksAll.forEach(link => {
  link.addEventListener("click", () => {
    navLinksAll.forEach(l => l.classList.remove("active"));
    link.classList.add("active");
  });
});

/* ================= Swiper Slide Cards ================= */

var BoxSlider = new Swiper('.slider', {
  effect: 'coverflow',
  grabCursor: true,
  centeredSlides: true,
  loop: true,
  slidesPerView: 'auto',

  coverflowEffect: {
    rotate: 0,
    stretch: 0,
    depth: 100,
    modifier: 2.5,
  },

  pagination: {
    el: '.swiper-pagination',
    clickable: true,
  },

  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  }
});


/* ================= Screen Loader ================= */

// const loadingMessages = [
//   "Initializing GestureX AI...",
//   "Loading Computer Vision...",
//   "Opening Camera...",
//   "Preparing FaceMesh...",
//   "Loading Tracking Engine...",
//   "Almost Ready..."
// ];

// let i = 0;
// const txt = document.getElementById("loadingText");
// const timer = setInterval(()=>{
//   i++;
//   if(i<loadingMessages.length){
//     txt.innerHTML=loadingMessages[i];
//   }
// },1300);

// window.onload=function(){
//   setTimeout(()=>{
//     clearInterval(timer);
//     document.getElementById("loader").style.opacity="0";
//     setTimeout(()=>{
//       document.getElementById("loader").style.display="none";
//     },600);
//   },7000);
// }
