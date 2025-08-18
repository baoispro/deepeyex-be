package handlers

import (
	"encoding/json"
	"net/http"
	"user-service/internal/services"
)

func RegisterHandler(w http.ResponseWriter, r *http.Request) {
	var body map[string]string
	json.NewDecoder(r.Body).Decode(&body)

	err := services.RegisterUser(body["email"], body["password"])
	if err != nil {
		http.Error(w, "❌ Register failed", http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	w.Write([]byte("✅ User registered"))
}

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	var body map[string]string
	json.NewDecoder(r.Body).Decode(&body)

	ok := services.LoginUser(body["email"], body["password"])
	if !ok {
		http.Error(w, "❌ Invalid credentials", http.StatusUnauthorized)
		return
	}

	w.Write([]byte("✅ Login success"))
}
