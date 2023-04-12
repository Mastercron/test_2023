const btn = document.getElementById("btn");
const para = document.getElementById("para");
let count = 0;
const responsesArr = [
  "Ha hecho clic en el botón tantas veces como: ",
  "Guau, te gusta hacer clic en ese botón. Las veces que has clicado en el botón son:",
  "¿Por qué sigues haciendo clic en él? Has aplicado en el botón los siguientes clics:",
  "Ahora solo estás siendo molesto. Las veces que has clicado son:"
];

btn.addEventListener("click", () => {
  count++;
  if (count < 10) {
    para.innerHTML = `${responsesArr[0]} ${count}`;
  } else if (count >= 10 && count < 15) {
    para.innerHTML = `${responsesArr[1]} ${count}`;
  } else if (count >= 15 && count < 20) {
    para.innerHTML = `${responsesArr[2]} ${count}`;
  } else {
    para.innerHTML = `${responsesArr[3]} ${count}`;
  }
});