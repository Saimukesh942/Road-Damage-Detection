/* RoadDamage Detection — Marketing Page JS */

document.addEventListener("DOMContentLoaded", () => {
  // Damage type tab switching
  const tabs = document.querySelectorAll(".damage-tab");
  const details = document.querySelectorAll(".damage-detail");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      details.forEach(d => d.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById("detail-" + tab.dataset.type);
      if (target) target.classList.add("active");
    });
  });
});
