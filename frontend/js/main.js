/* ==========================================================================
   StrokeGuard — Shared Navigation (Vanilla JS)
   ========================================================================== */

(function () {
  const toggle = document.querySelector('.nav__toggle');
  const navList = document.getElementById('nav-list');

  if (!toggle || !navList) return;

  toggle.addEventListener('click', function () {
    const open = navList.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open === true ? 'true' : 'false');
  });

  // Close mobile menu when a nav link is clicked
  navList.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () {
      navList.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });

  // Highlight the current page in navigation
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  navList.querySelectorAll('a').forEach(function (link) {
    const href = link.getAttribute('href');
    if (href && currentPath === href.split('/').pop()) {
      link.setAttribute('aria-current', 'page');
      link.classList.add('is-current');
    }
  });
})();
