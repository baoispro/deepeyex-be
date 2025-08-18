package router

import (
	"github.com/go-chi/chi/v5"
	"user-service/internal/handlers"
)

func SetupRouter() *chi.Mux {
	r := chi.NewRouter()

	r.Post("/register", handlers.RegisterHandler)
	r.Post("/login", handlers.LoginHandler)

	return r
}
