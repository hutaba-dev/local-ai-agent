const loginForm = document.querySelector("#login-form");
const loginError = document.querySelector("#login-error");

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.textContent = "";
  const response = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: loginForm.username.value, password: loginForm.password.value }),
  });
  if (!response.ok) {
    loginError.textContent = "Sign in failed.";
    return;
  }
  location.assign("/");
});