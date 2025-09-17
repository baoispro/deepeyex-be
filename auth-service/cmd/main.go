package main

import (
	"auth-service/internal/config"
	"auth-service/internal/database"
	"auth-service/internal/handlers"
	"auth-service/internal/repositories"
	"auth-service/internal/routers"
	"auth-service/internal/services"
	"log"

	_ "auth-service/docs"
)

// @title Auth Service API
// @version 1.0
// @BasePath /
// @host localhost:8080
func main() {
	cfg := config.Load()

	db := database.Connect(cfg)
	if err := database.AutoMigrate(db); err != nil {
		log.Fatal(err)
	}

	userRepo := repositories.NewUserRepo(db)
	tokenRepo := repositories.NewTokenRepo(db)
	authService := services.NewAuthService(cfg, userRepo, tokenRepo)
	userService := services.NewUserService(userRepo)
	authHandler := handlers.NewAuthHandler(cfg, authService)
	userHandler := handlers.NewUserHandler(userService)

	r := routers.SetupRouter(&cfg, authHandler, userHandler)
	log.Printf("Auth service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
