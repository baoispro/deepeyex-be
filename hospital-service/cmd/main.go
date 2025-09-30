package main

import (
	"log"

	_ "hospital-service/docs"
	"hospital-service/internal/config"
	"hospital-service/internal/database"
	"hospital-service/internal/handlers/bookinghandler"
	patienthandler "hospital-service/internal/handlers/patienthandler"
	patientrepo "hospital-service/internal/repositories/patientrepo"
	"hospital-service/internal/services/bookingservice"
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

	// MedicalRecord
	medicalrecordhandler "hospital-service/internal/handlers/medicalrecordhandler"
	medicalrecordrepo "hospital-service/internal/repositories/medicalrecordrepo"
	medicalrecordservice "hospital-service/internal/services/medicalrecordservice"

	// Prescription
	prescriptionhandler "hospital-service/internal/handlers/medicalrecordhandler"
	prescriptionrepo "hospital-service/internal/repositories/medicalrecordrepo" // cùng repo với prescription
	prescriptionservice "hospital-service/internal/services/medicalrecordservice"

	// PrescriptionItem
	prescriptionitemhandler "hospital-service/internal/handlers/medicalrecordhandler"
	prescriptionitemrepo "hospital-service/internal/repositories/medicalrecordrepo" // cùng repo với prescription
	prescriptionitemservice "hospital-service/internal/services/medicalrecordservice"

	// Attachment
	attachmenthandler "hospital-service/internal/handlers/medicalrecordhandler"
	attachmentrepo "hospital-service/internal/repositories/medicalrecordrepo"
	attachmentservice "hospital-service/internal/services/medicalrecordservice"

	// FollowUp
	followuphandler "hospital-service/internal/handlers/medicalrecordhandler"
	followuprepo "hospital-service/internal/repositories/medicalrecordrepo"
	followupservice "hospital-service/internal/services/medicalrecordservice"

	// Service
	servicehandler "hospital-service/internal/handlers/servicehandler"
	servicerepo "hospital-service/internal/repositories/servicerepo"
	doctorserviceservice "hospital-service/internal/services/doctorserviceservice"

	// AuditTrail

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
	medicalRecordRepo := medicalrecordrepo.NewMedicalRecordRepository(db)
	prescriptionRepo := prescriptionrepo.NewPrescriptionRepository(db)
	attachmentRepo := attachmentrepo.NewAttachmentRepository(db)
	followUpRepo := followuprepo.NewFollowUpRepository(db)
	prescriptionitemrepo := prescriptionitemrepo.NewPrescriptionItemRepository(db)
	serviceRepo := servicerepo.NewServiceRepo(db)

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
	aService := appointmentservice.NewAppointmentService(aRepo, tRepo)
	tService := timeslotservice.NewTimeSlotService(tRepo)
	drugService := drugservice.NewDrugService(drugRepo, s3Client)
	orderService := orderservice.NewOrderService(orderRepo, drugRepo, s3Client)
	serviceService := doctorserviceservice.NewServiceService(serviceRepo)

	medicalRecordService := medicalrecordservice.NewMedicalRecordService(medicalRecordRepo)
	prescriptionService := prescriptionservice.NewPrescriptionService(prescriptionRepo)
	attachmentService := attachmentservice.NewAttachmentService(attachmentRepo)
	followUpService := followupservice.NewFollowUpService(followUpRepo)
	prescriptionItemService := prescriptionitemservice.NewPrescriptionItemService(prescriptionitemrepo)
	bookingService := bookingservice.NewBookingService(aService, orderService)

	// Initialize handlers
	pHandler := patienthandler.NewPatientHandler(cfg, pService)
	dHandler := doctorhandler.NewDoctorHandler(cfg, dService)
	hHandler := hospitalhandler.NewHospitalHandler(cfg, hService)
	aHandler := appointmenthandler.NewAppointmentHandler(cfg, aService)
	tHandler := timeslothandler.NewTimeSlotHandler(cfg, tService)
	drugHandler := drughandler.NewDrugHandler(cfg, drugService)
	orderHandler := orderhandler.NewOrderHandler(cfg, orderService)
	medicalRecordHandler := medicalrecordhandler.NewMedicalRecordHandler(cfg, medicalRecordService)
	prHandler := prescriptionhandler.NewPrescriptionHandler(cfg, prescriptionService)
	attachmentHandler := attachmenthandler.NewAttachmentHandler(cfg, attachmentService)
	followUpHandler := followuphandler.NewFollowUpHandler(cfg, followUpService)
	prescriptionItemHander := prescriptionitemhandler.NewPrescriptionItemHandler(cfg, prescriptionItemService)
	bookingHandler := bookinghandler.NewBookingHandler(bookingService)
	serviceHandler := servicehandler.NewServiceHandler(cfg, serviceService)

	// Setup router
	r := routers.SetupRouter(&cfg, pHandler, dHandler, hHandler, aHandler, tHandler, drugHandler, orderHandler, medicalRecordHandler, prHandler, attachmentHandler, followUpHandler, prescriptionItemHander,serviceHandler, bookingHandler )

	log.Printf("Hospital service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
