var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
} 

/**
 * Sorts a HTML table.
 *
 * @param {HTMLTableElement} table The table to sort
 * @param {number} column The index of the column to sort
 * @param {boolean} asc Determines if the sorting will be in ascending
 */
function sortTableByColumn(table, column, asc = true) {
	const dirModifier = asc ? 1 : -1;
	const tBody = table.tBodies[0];
	const rows = Array.from(tBody.querySelectorAll("tr"));

	// Sort each row
	const sortedRows = rows.sort((a, b) => {
		const aColText = a.querySelector(`td:nth-child(${column + 1})`).textContent.trim();
		const bColText = b.querySelector(`td:nth-child(${column + 1})`).textContent.trim();

		return aColText > bColText ? (1 * dirModifier) : (-1 * dirModifier);
	});

	// Remove all existing TRs from the table
	while (tBody.firstChild) {
		tBody.removeChild(tBody.firstChild);
	}

	// Re-add the newly sorted rows
	tBody.append(...sortedRows);

	// Remember how the column is currently sorted
	table.querySelectorAll("th").forEach(th => th.classList.remove("th-sort-asc", "th-sort-desc"));
	table.querySelector(`th:nth-child(${column + 1})`).classList.toggle("th-sort-asc", asc);
	table.querySelector(`th:nth-child(${column + 1})`).classList.toggle("th-sort-desc", !asc);
}

document.querySelectorAll(".table-sortable th").forEach(headerCell => {
	headerCell.addEventListener("click", () => {
		const tableElement = headerCell.parentElement.parentElement.parentElement;
		const headerIndex = Array.prototype.indexOf.call(headerCell.parentElement.children, headerCell);
		const currentIsAscending = headerCell.classList.contains("th-sort-asc");

		sortTableByColumn(tableElement, headerIndex, !currentIsAscending);
	});
});

/* Source: https://codepen.io/hbuchel/pen/jOGbGE
$('.button, .close').on('click', function(e) {
    e.preventDefault();
    $('.detail, html, body').toggleClass('open');
  });

  document.getElementById('retrain-form').addEventListener('submit', function (event) {
	event.preventDefault();

	// Show "Retraining..." notification
	showNotification('Retraining in progress...', 'info');

	fetch(this.action, {
		method: this.method,
		body: new FormData(this),
		headers: {
			'X-CSRFToken': document.getElementsByName('csrfmiddlewaretoken')[0].value
		}
	})
	.then(response => response.text())
	.then(result => {
		// Hide the previous notification
		hideNotification();

		// Show success 
		showNotification('Retraining successful!', 'success');

		console.log('Result:', result);
	})
	.catch(error => {
		// Hide the previous notification
		hideNotification();

		// Show error notification
		showNotification('Error during retraining!', 'error');
		console.error('Error:', error);
	});
});

function showNotification(message, type) {
	alert(message);
}

function hideNotification() {
	// Add if needed
}*/
