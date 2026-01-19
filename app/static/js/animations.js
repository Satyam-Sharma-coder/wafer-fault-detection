document.addEventListener("DOMContentLoaded", () => {
  const btns = document.querySelectorAll(".button");
  btns.forEach(b => b.addEventListener("mouseenter", () => b.style.opacity = "0.85"));
  btns.forEach(b => b.addEventListener("mouseleave", () => b.style.opacity = "1.0"));
});