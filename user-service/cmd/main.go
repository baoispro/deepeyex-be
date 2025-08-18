package main

import (
	"log"
	"net/http"
	"user-service/config"
	"user-service/internal/database"
	"user-service/internal/router"
)

func main() {
	cfg := config.LoadConfig()
	database.Connect(cfg)

	r := router.SetupRouter()

	log.Println("🚀 Server running on port", cfg.ServerPort)
	http.ListenAndServe(":"+cfg.ServerPort, r)
}
