function Usuario(usuario, nombre, email, password) {
  this.usuario = usuario;
  this.nombre = nombre;
  this.email = email;
  this.password = password;
}

Usuario.prototype.saludar = function() {
  return `Hola, mi nombre es ${this.nombre}`;
}

Usuario.prototype.datos = function() {
  return `${this.nombre}, ${this.email}, ${this.password}`;
}

// Ejemplo de uso:
const miUsuario = new Usuario("miusuario", "Juan", "juan@example.com", "password123");
console.log(miUsuario.saludar()); // "Hola, mi nombre es Juan"
console.log(miUsuario.datos()); // "Juan, juan@example.com, password123"