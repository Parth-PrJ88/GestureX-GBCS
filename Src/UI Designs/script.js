/* Set scroll padding to navigation height */
// const navigation = document.querySelector(".primary-navigation");
// const navigationHeight = navigation.offsetheight;
// document.documentElement.style.setProperty("--scroll-padding", navigationHeight + "px");

/* Initialize Swiper */
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
});slider-control