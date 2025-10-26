package main

import (
	"log"

	_ "hospital-service/docs"
	"hospital-service/internal/config"
	"hospital-service/internal/database"
	"hospital-service/internal/handlers/bookinghandler"
	"hospital-service/internal/handlers/callhandler"
	"hospital-service/internal/handlers/fullrecordhandler"
	"hospital-service/internal/handlers/notificationhandler"
	patienthandler "hospital-service/internal/handlers/patienthandler"
	"hospital-service/internal/handlers/paymenthandler"
	"hospital-service/internal/handlers/uploadhandler"
	"hospital-service/internal/repositories/notificationrepo"
	patientrepo "hospital-service/internal/repositories/patientrepo"
	"hospital-service/internal/services/bookingservice"
	"hospital-service/internal/services/fullrecordservice"
	"hospital-service/internal/services/notificationservice"
	patientservice "hospital-service/internal/services/patientservice"
	"hospital-service/internal/services/paymentservice"
	"hospital-service/internal/services/uploadservice"
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

	// Service
	servicehandler "hospital-service/internal/handlers/servicehandler"
	servicerepo "hospital-service/internal/repositories/servicerepo"
	doctorserviceservice "hospital-service/internal/services/doctorserviceservice"

	// Cron Service
	cronservice "hospital-service/internal/services/cronservice"

	// Email Service
	emailhandler "hospital-service/internal/handlers/emailhandler"
	emailservice "hospital-service/internal/services/emailservice"

	// WebSocket
	websockethandler "hospital-service/internal/handlers/websockethandler"
	"hospital-service/internal/websocket"

	"hospital-service/internal/routers"
)

// @title Auth Service API
// @version 1.0
// @BasePath /
// @host localhost:8084
func main() {
	// Load configuration
	cfg := config.Load()

	// Connect database
	db := database.Connect(cfg)
	if err := database.AutoMigrate(db); err != nil {
		log.Fatal(err)
	}

	// ✅ Initialize WebSocket Hub
	wsHub := websocket.NewHub()
	go wsHub.Run() // Start hub in goroutine
	log.Println("WebSocket Hub started successfully")

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
	prescriptionitemrepo := prescriptionitemrepo.NewPrescriptionItemRepository(db)
	serviceRepo := servicerepo.NewServiceRepo(db)
	aidiagnosisRepo := medicalrecordrepo.NewAIDiagnosisRepo(db)
	medicationReminderRepo := medicalrecordrepo.NewMedicationReminderRepository(db)
	notificationRepo := notificationrepo.NewNotificationRepo(db)

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
	notificationService := notificationservice.NewNotificationService(notificationRepo, wsHub)
	pService := patientservice.NewPatientService(pRepo, s3Client)
	dService := doctorservice.NewDoctorService(dRepo, s3Client)
	hService := hospitalservice.NewHospitalService(hRepo, s3Client)
	aService := appointmentservice.NewAppointmentService(aRepo, tRepo, dRepo)
	tService := appointmentservice.NewTimeSlotService(tRepo, dRepo, aRepo)
	drugService := drugservice.NewDrugService(drugRepo, s3Client)
	orderService := orderservice.NewOrderService(orderRepo, drugRepo, s3Client)
	serviceService := doctorserviceservice.NewServiceService(serviceRepo)
	medicalRecordService := medicalrecordservice.NewMedicalRecordService(medicalRecordRepo, aidiagnosisRepo)
	prescriptionService := prescriptionservice.NewPrescriptionService(prescriptionRepo, prescriptionitemrepo, medicationReminderRepo)
	attachmentService := attachmentservice.NewAttachmentService(attachmentRepo, s3Client)
	prescriptionItemService := prescriptionitemservice.NewPrescriptionItemService(prescriptionitemrepo)
	bookingService := bookingservice.NewBookingService(aService, orderService, wsHub, notificationService) // ✅ Pass WebSocket Hub
	vnpayService := paymentservice.NewVnpayService(cfg)
	emailService := emailservice.NewEmailService(cfg)
	uploadservice := uploadservice.NewUploadService(s3Client)
	aidiagnosisService := medicalrecordservice.NewAIDiagnosisService(aidiagnosisRepo, s3Client)
	fullRecordService := fullrecordservice.NewFullRecordService(medicalRecordService, attachmentService, prescriptionService, aService)

	// Initialize cron service
	cronService := cronservice.NewCronService(tService)

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
	prescriptionItemHander := prescriptionitemhandler.NewPrescriptionItemHandler(cfg, prescriptionItemService)
	bookingHandler := bookinghandler.NewBookingHandler(bookingService)
	serviceHandler := servicehandler.NewServiceHandler(cfg, serviceService)
	vnpayHandler := paymenthandler.NewVnpayHandler(vnpayService)
	emailHandler := emailhandler.NewEmailHandler(emailService)
	uploadhandler := uploadhandler.NewUploadHandler(uploadservice)
	callhandler := callhandler.NewStringeeHandler()
	wsHandler := websockethandler.NewWebSocketHandler(wsHub) // ✅ WebSocket Handler
	aidiagnosisHandler := medicalrecordhandler.NewAIDiagnosisHandler(aidiagnosisService)
	fullRecordHandler := fullrecordhandler.NewFullRecordHandler(fullRecordService)
	notificationHandler := notificationhandler.NewNotificationHandler(notificationService)

	// Start cron service
	if err := cronService.Start(); err != nil {
		log.Printf("Error starting cron service: %v", err)
	} else {
		log.Println("Cron service started successfully - Will run every Saturday at 23:00")
	}

	// Setup router
	r := routers.SetupRouter(&cfg, pHandler, dHandler, hHandler, aHandler, tHandler, drugHandler, orderHandler, medicalRecordHandler, prHandler, attachmentHandler, prescriptionItemHander, serviceHandler, bookingHandler, vnpayHandler, emailHandler, uploadhandler, callhandler, wsHandler, aidiagnosisHandler, fullRecordHandler, notificationHandler)

	log.Printf("Hospital service running on :%s", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal(err)
	}
}
