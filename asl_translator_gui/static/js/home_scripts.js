/* Source: https://codepen.io/anithvishwanath/pen/qYrWJJ */
$("#input_frame").css("opacity", "0");

$("#file_browser").click(function(e) {
  e.preventDefault();
  $("#input_file").trigger("click");
});
