/* Source: https://codepen.io/hbuchel/pen/jOGbGE */
$('.button, .close').on('click', function(e) {
    e.preventDefault();
    $('.detail, html, body').toggleClass('open');
  });