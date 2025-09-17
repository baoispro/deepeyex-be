package main

import (
	"log"

	_ "hospital-service/docs"
	"hospital-service/internal/config"
	"hospital-service/internal/database"
	patienthandler "hospital-service/internal/handlers/patienthandler"
	patientrepo "hospital-service/internal/repositories/patientrepo"
	patientservice "hospital-service/internal/services/patientservice"
	"hospital-service/internal/storage"

	doctorhandler "hospital-service/internal/handlers/doctorhandler"
	doctorrepo "hospital-service/internal/repositories/doctorrepo"
	doctorservice "hospital-service/internal/services/doctorservice"

	hospitalhandler "hospital-service/internal/handlers/hospitalhandler"
	hospitalrepo "hospital-service/internal/repositories/hospitalrepo"
	hospitalservice "hospital-service/internal/services/hospitalservice"

	appointmenthandler "hospital-service/internal/handlers/appointmenthandler"
	appointmentrepo "hospital-service/internal/repositories/appointmentrepo"
	appointmentservice "hospital-service/internal/services/appointmentservice"

	timeslothandler "hospital-service/internal/handlers/appointmenthandler"
	timeslotrepo "hospital-service/internal/repositories/appointmentrepo"
	timeslotservice "hospital-service/internal/services/appointmentservice"

	drughandler "hospital-service/internal/handlers/drughandler"
	drugrepo "hospital-service/internal/repositories/drugrepo"
	drugservice "hospital-service/internal/services/drugservice"

	orderhandler "hospital-service/internal/handlers/orderhandler"
	orderrepo "hospital-service/internal/repositories/orderrepo"
	orderservice "hospital-service/internal/services/orderservice"

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
	aRepo := appointmentrepo.NewAppointmentRepo(db)
	tRepo := timeslotrepo.NewTimeSlotRepo(db)
	drugRepo := drugrepo.NewDrugRepo(db)
	orderRepo := orderrepo.NewOrderRepo(db)

	s3Client, err := storage.NewS3Client(
		cfg.S3Bucket,
		cfg.S3Region,
		cfg.AWSAccessKey,
		cfg.AWSSecretKey,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Initialize services
	pService := patientservice.NewPatientService(pRepo, s3Client)
	dService := doctorservice.NewDoctorService(dRepo, s3Client)
	hService := hospitalservice.NewHospitalService(hRepo, s3Client)
	aService := appointmentservice.NewAppointmentService(aRepo)
	tService := timeslotservice.NewTimeSlotService(tRepo)
	drugService := drugservice.NewDrugService(drugRepo, s3Client)
	orderService := orderservice.NewOrderService(orderRepo, drugRepo, s3Client)

	// Initialize handlers
	pHandler := patienthandler.NewPatientHandler(cfg, pService)
	dHandler := doctorhandler.NewDoctorHandler(cfg, dService)
	hHandler := hospitalhandler.NewHospitalHandler(cfg, hService)
	aHandler := appointmenthandler.NewAppointmentHandler(cfg, aService)
	tHandler := timeslothandler.NewTimeSlotHandler(cfg, tService)
	drugHandler := drughandler.NewDrugHandler(cfg, drugService)
	orderHandler := orderhandler.NewOrderHandler(cfg, orderService)

	// Setup router
	r := routers.SetupRouter(&cfg, pHandler, dHandler, hHandler, aHandler, tHandler, drugHandler, orderHandler)

	log.Printf("Hospital service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
