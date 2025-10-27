document.getElementById('predictForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const response = await fetch('/predict', { method: 'POST', body: formData });
  const data = await response.json();

  const resultBox = document.getElementById('result');
  const price = data.estimated_price;

  // Reset result visibility
  resultBox.classList.remove('show');
  resultBox.innerHTML = `<h3>Estimated Price: ₹ <span id="price">0</span> Lakhs</h3>`;

  setTimeout(() => resultBox.classList.add('show'), 100);

  // Animate number counting up
  let current = 0;
  const duration = 1200; // ms
  const stepTime = 15;
  const increment = price / (duration / stepTime);
  const priceEl = document.getElementById('price');

  const timer = setInterval(() => {
    current += increment;
    if (current >= price) {
      current = price;
      clearInterval(timer);
    }
    priceEl.textContent = current.toFixed(2);
  }, stepTime);
});