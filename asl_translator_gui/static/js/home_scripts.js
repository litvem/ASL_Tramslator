// Select all elements
const dropArea = document.querySelector(".drag-area");

// Global variable to be used for uploading file
let file;

// If user drags file over DropArea
dropArea.addEventListener("dragover", (event) => {
  event.preventDefault(); // Prevent from default behaviour
  dropArea.classList.add("active");
});

// If user drags file away from DropArea
dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("active");
});

// If user drops file in DropArea
dropArea.addEventListener("drop", (event) => {
  event.preventDefault(); // Prevent from default behaviour
  // If user select multiple files, only the first one will be used
  file = event.dataTransfer.files[0];
  let fileType = file.type;
  let validExtensions =["video/mp4"]; // Restriction for uploading file format
  if(validExtensions.includes(fileType) && file.size < 104857600) { // If user selects .mp4 format file and it's smaller that 100MB
    let fileReader = new FileReader();  // Create new FileReader object
    fileReader.onload = () => {
      let fileURL = fileReader.result;  // Pass user file source to fileURL variable
      // TODO add here code to pass file to the model for processing
      console.log(fileType); 
    }
    fileReader.readAsDataURL(file);
  } else { // Is user selects file with another format
    alert("Only .mp4 files not exceeding 100MB are permitted. Try again.");
    dropArea.classList.remove("active");
  }
});
