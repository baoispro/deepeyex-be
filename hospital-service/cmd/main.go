package main

import (
	"log"

	_ "hospital-service/docs"
	"hospital-service/internal/config"
	"hospital-service/internal/database"
	patienthandler "hospital-service/internal/handlers/patienthandler"
	patientrepo "hospital-service/internal/repositories/patientrepo"
	patientservice "hospital-service/internal/services/patientservice"

	doctorhandler "hospital-service/internal/handlers/doctorhandler"
	doctorrepo "hospital-service/internal/repositories/doctorrepo"
	doctorservice "hospital-service/internal/services/doctorservice"

	hospitalhandler "hospital-service/internal/handlers/hospitalhandler"
	hospitalrepo "hospital-service/internal/repositories/hospitalrepo"
	hospitalservice "hospital-service/internal/services/hospitalservice"

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
	dRepo := doctorrepo.NewDoctorRepo(db)
	hRepo := hospitalrepo.NewHospitalRepo(db)

	// Initialize services
	pService := patientservice.NewPatientService(pRepo)
	dService := doctorservice.NewDoctorService(dRepo)
	hService := hospitalservice.NewHospitalService(hRepo)

	// Initialize handlers
	pHandler := patienthandler.NewPatientHandler(cfg,pService)
	dHandler := doctorhandler.NewDoctorHandler(cfg, dService)
	hHandler := hospitalhandler.NewHospitalHandler(cfg, hService)

	// Setup router
	r := routers.SetupRouter(&cfg, pHandler, dHandler, hHandler)

	log.Printf("Hospital service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
