package main

import (
	"log"

	_ "hospital-service/docs"
	"hospital-service/internal/config"
	"hospital-service/internal/database"
	patienthandler "hospital-service/internal/handlers/patienthandler"
	patientrepo "hospital-service/internal/repositories/patientrepo"
	patientservice "hospital-service/internal/services/patientservice"
	"hospital-service/internal/routers"
)

// @title Auth Service API
// @version 1.0
// @BasePath /
// @host localhost:8081
func main() {
	// Load configuration
	cfg := config.Load()

	// Connect database
	db := database.Connect(cfg)
	if err := database.AutoMigrate(db); err != nil {
		log.Fatal(err)
	}

	// Initialize repositories
	pRepo := patientrepo.NewPatientRepo(db)

	// Initialize services
	pService := patientservice.NewPatientService(pRepo)

	// Initialize handlers
	pHandler := patienthandler.NewPatientHandler(cfg,pService)

	// Setup router
	r := routers.SetupRouter(&cfg, pHandler)

	log.Printf("Hospital service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
