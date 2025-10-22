package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/appointmenthandler"
	"hospital-service/internal/handlers/bookinghandler"
	"hospital-service/internal/handlers/callhandler"
	"hospital-service/internal/handlers/doctorhandler"
	"hospital-service/internal/handlers/drughandler"
	"hospital-service/internal/handlers/emailhandler"
	"hospital-service/internal/handlers/fullrecordhandler"
	"hospital-service/internal/handlers/hospitalhandler"
	"hospital-service/internal/handlers/medicalrecordhandler"
	"hospital-service/internal/handlers/notificationhandler"
	"hospital-service/internal/handlers/orderhandler"
	"hospital-service/internal/handlers/patienthandler"
	"hospital-service/internal/handlers/paymenthandler"
	"hospital-service/internal/handlers/servicehandler"
	"hospital-service/internal/handlers/uploadhandler"
	"hospital-service/internal/handlers/websockethandler"

	"hospital-service/internal/middlewares"

	"github.com/gin-contrib/cors"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler, doctorHandler *doctorhandler.DoctorHandler, hHandler *hospitalhandler.HospitalHandler, aHandler *appointmenthandler.AppointmentHandler, tHandler *appointmenthandler.TimeSlotHandler, drugHandler *drughandler.DrugHandler, orderHandler *orderhandler.OrderHandler, medicalRecordHandler *medicalrecordhandler.MedicalRecordHandler,
	prescriptionHandler *medicalrecordhandler.PrescriptionHandler,
	attachmentHandler *medicalrecordhandler.AttachmentHandler,
	prescriptionItemHandler *medicalrecordhandler.PrescriptionItemHandler,
	serviceHandler *servicehandler.ServiceHandler,
	bookingHandler *bookinghandler.BookingHandler,
	vnpayHandler *paymenthandler.VnpayHandler,
	emailHandler *emailhandler.EmailHandler,
	uploadhandler *uploadhandler.UploadHandler,
	callhandler *callhandler.StringeeHandler,
	wsHandler *websockethandler.WebSocketHandler,
	aidiagnosisHandler *medicalrecordhandler.AIDiagnosisHandler,
	fullRecordHandler *fullrecordhandler.FullRecordHandler,
	handler *notificationhandler.NotificationHandler,
) *gin.Engine {
	r := gin.Default()

	r.Use(cors.Default())
	r.Use(middlewares.LimitRequestBody(5 << 20))

	// ===== Patient routes =====
	patient := r.Group("/patients")
	{
		patient.POST("", patientHandler.CreatePatient)                   // Create
		patient.GET("", patientHandler.ListPatients)                     // List all
		patient.GET("/user/:user_id", patientHandler.GetPatientByUserID) // Get by UserID
		patient.GET("/:patient_id", patientHandler.GetPatientByID)       // Get by PatientID
		patient.PUT("/:patient_id", patientHandler.UpdatePatient)        // Update
		patient.DELETE("/:patient_id", patientHandler.DeletePatient)     // Delete
	}

	// ===== Doctor routes =====
	doctor := r.Group("/doctors")
	{
		doctor.POST("", doctorHandler.CreateDoctor)                   // Create
		doctor.GET("", doctorHandler.ListDoctors)                     // List all
		doctor.GET("/user/:user_id", doctorHandler.GetDoctorByUserID) // Get by UserID
		doctor.GET("/slug/:slug", doctorHandler.GetDoctorBySlug)
		doctor.GET("/hospital/:hospital_id", doctorHandler.ListDoctorsByHospitalID) // List doctors by hospital_id
		doctor.GET("/:doctor_id", doctorHandler.GetDoctorByID)                      // Get by DoctorID
		doctor.PUT("/:doctor_id", doctorHandler.UpdateDoctor)                       // Update
		doctor.DELETE("/:doctor_id", doctorHandler.DeleteDoctor)                    // Delete
	}

	// ===== Hospital routes =====
	hospital := r.Group("/hospitals")
	{
		hospital.GET("/slug/:slug", hHandler.GetHospitalBySlug)
		hospital.GET("/cities", hHandler.ListCities)
		hospital.GET("/wards", hHandler.ListWardsByCity)
		hospital.GET("/search/address", hHandler.SearchByAddress)
		hospital.GET("/filter", hHandler.ListByCityAndWard)
		hospital.POST("/nearby", hHandler.FindNearbyHospitals)
		hospital.POST("", hHandler.CreateHospital)
		hospital.GET("", hHandler.ListHospitals)
		hospital.GET("/:hospital_id", hHandler.GetHospitalByID)
		hospital.PUT("/:hospital_id", hHandler.UpdateHospital)
		hospital.DELETE("/:hospital_id", hHandler.DeleteHospital)
	}

	// ===== Appointments routes =====
	appointments := r.Group("/appointments")
	{
		appointments.GET("/:appointment_id", aHandler.GetAppointmentByID)
		appointments.GET("/patient/:patient_id", aHandler.GetAppointmentsByPatient)
		appointments.GET("/doctor/:doctor_id", aHandler.GetAppointmentsByDoctor)
		appointments.POST("/follow-up", aHandler.CreateFollowUpAppointment)
		appointments.PUT("/:appointment_id/status", aHandler.UpdateAppointmentStatus)
		appointments.PUT("/:appointment_id/detail", aHandler.UpdateAppointmentDetail)
		appointments.PUT("/:appointment_id/cancel", aHandler.CancelAppointment)
		appointments.GET("", aHandler.ListAllAppointments)
		appointments.GET("/online", aHandler.GetOnlineAppointments)
		appointments.GET("/today", aHandler.GetTodayAppointments)
		appointments.DELETE("/:appointment_id", aHandler.DeleteAppointment)
	}

	// ===== Timeslots routes =====
	timeSlot := r.Group("/timeslots")
	{
		timeSlot.POST("", tHandler.CreateTimeSlot)
		timeSlot.POST("/batch", tHandler.CreateBatch)
		timeSlot.POST("/multi-shift", tHandler.CreateMultiShift)
		timeSlot.POST("/import-dayoff", tHandler.ImportDoctorDayOff)
		timeSlot.GET("", tHandler.ListAllTimeSlots)
		timeSlot.GET("/:slot_id", tHandler.GetTimeSlotByID)
		timeSlot.GET("/doctor/:doctor_id", tHandler.GetTimeSlotsByDoctor)
		timeSlot.GET("/doctor/:doctor_id/date", tHandler.GetTimeSlotsByDoctorAndDate)
		timeSlot.GET("/doctor/:doctor_id/month", tHandler.GetTimeSlotsByDoctorAndMonth)
		timeSlot.GET("/doctor/:doctor_id/date-range", tHandler.GetTimeSlotsByDoctorAndDateRange)
		timeSlot.PUT("/:slot_id", tHandler.UpdateTimeSlot)
		timeSlot.DELETE("/:slot_id", tHandler.DeleteTimeSlot)

	}

	// ===== Drug routes =====
	drug := r.Group("/drugs")
	{
		drug.POST("", drugHandler.CreateDrug)            // Create
		drug.GET("", drugHandler.ListDrugs)              // List all
		drug.GET("/:drug_id", drugHandler.GetDrugByID)   // Get by DrugID
		drug.PUT("/:drug_id", drugHandler.UpdateDrug)    // Update
		drug.DELETE("/:drug_id", drugHandler.DeleteDrug) // Delete
	}

	// ===== Order routes =====
	order := r.Group("/orders")
	{
		order.POST("", orderHandler.CreateOrder)                           // Create order
		order.GET("", orderHandler.ListAllOrders)                          // List all
		order.GET("/:order_id", orderHandler.GetOrderByID)                 // Get by OrderID
		order.GET("/patient/:patient_id", orderHandler.GetOrdersByPatient) // Get orders by patient
		order.PUT("/:order_id/status", orderHandler.UpdateOrderStatus)     // Update order status
		order.PUT("/:order_id/appointment", orderHandler.UpdateOrderAppointment)
		order.DELETE("/:order_id", orderHandler.DeleteOrder) // Delete order
	}

	// ===== MedicalRecord routes =====
	medical := r.Group("/medical_records")
	{
		medical.POST("", medicalRecordHandler.CreateMedicalRecord) // Create
		medical.GET("", medicalRecordHandler.ListMedicalRecords)
		medical.GET("/check", medicalRecordHandler.CheckMedicalRecord)
		medical.GET("/patient", medicalRecordHandler.GetRecordsByPatient)
		medical.GET("/:record_id", medicalRecordHandler.GetMedicalRecord)    // Get by ID
		medical.PUT("/:record_id", medicalRecordHandler.UpdateMedicalRecord) // Update
		medical.DELETE("/:record_id", medicalRecordHandler.DeleteMedicalRecord)
		medical.POST("/init", medicalRecordHandler.InitMedicalRecordAndDiagnosis)
	}

	// ===== Prescription routes =====
	prescription := r.Group("/prescriptions")
	{
		// prescription.POST("", prescriptionHandler.CreatePrescription)
		prescription.GET("/patient/:patient_id", prescriptionHandler.GetPrescriptionsByPatientID)
		prescription.GET("/medical_records/:record_id", prescriptionHandler.ListPrescriptionsByMedicalRecordID)
		prescription.GET("/:prescription_id", prescriptionHandler.GetPrescriptionByID)
		prescription.PUT("/:prescription_id", prescriptionHandler.UpdatePrescription)
		prescription.PUT("/:prescription_id/approve", prescriptionHandler.ApprovePrescription)
		prescription.DELETE("/:prescription_id", prescriptionHandler.DeletePrescription)
	}

	// ===== Attachment routes =====
	attachment := r.Group("/attachments")
	{
		attachment.POST("", attachmentHandler.AddAttachment)
		attachment.GET("/:record_id/medical_records", attachmentHandler.GetAttachments)
		attachment.DELETE("/:id", attachmentHandler.DeleteAttachment)
	}

	// ===== Service routes =====
	services := r.Group("/services")
	{
		services.POST("", serviceHandler.CreateService)                // Create service
		services.GET("", serviceHandler.ListAllServices)               // List all services
		services.GET("/:service_id", serviceHandler.GetServiceByID)    // Get service by ID
		services.PUT("/:service_id", serviceHandler.UpdateService)     // Update service
		services.DELETE("/:service_id", serviceHandler.DeleteService)  // Delete service
		services.POST("/assign", serviceHandler.AssignServiceToDoctor) // Assign service to doctor
	}

	// ===== Doctor-Service routes =====
	doctorServices := r.Group("/doctors/:doctor_id/services")
	{
		doctorServices.GET("", serviceHandler.ListServicesByDoctorID)                 // List services by doctor
		doctorServices.DELETE("/:service_id", serviceHandler.RemoveServiceFromDoctor) // Remove service from doctor
	}

	// ===== Booking routes =====
	booking := r.Group("/bookings")
	{
		booking.POST("", bookingHandler.CreateBooking)
	}

	// ===== Payment routes =====
	vnpay := r.Group("/vnpay")
	{
		vnpay.POST("/create-payment", vnpayHandler.CreatePayment)
		vnpay.GET("/return", vnpayHandler.VnpayReturn)
	}

	// ===== Email routes =====
	email := r.Group("/emails")
	{
		email.POST("/send", emailHandler.SendEmail)
		email.POST("/appointment-confirmation", emailHandler.SendAppointmentConfirmation)
		email.POST("/appointment-reminder", emailHandler.SendAppointmentReminder)
		email.POST("/prescription", emailHandler.SendPrescription)
		email.POST("/order-confirmation", emailHandler.SendOrderConfirmation)
	}

	// ===== Upload routes =====
	upload := r.Group("/upload")
	{
		upload.POST("", uploadhandler.UploadFile)
	}

	// ===== Call routes =====
	call := r.Group("/call")
	{
		call.GET("stringee-token", callhandler.GetStringeeToken)
	}

	// ===== WebSocket routes =====
	ws := r.Group("/ws")
	{
		ws.GET("", wsHandler.ServeWS)                                     // WebSocket connection endpoint
		ws.GET("/connected", wsHandler.GetConnectedDoctors)               // Get list of connected doctors
		ws.GET("/status/:doctor_id", wsHandler.GetDoctorConnectionStatus) // Check doctor connection status
		// ====== Patient WebSocket ======
		ws.GET("/patient", wsHandler.ServeWSPatient)                                // WebSocket connection endpoint (patient)
		ws.GET("/patient/connected", wsHandler.GetConnectedPatients)                // Get list of connected patients
		ws.GET("/patient/status/:patient_id", wsHandler.GetPatientConnectionStatus) // Check patient connection status
	}

	// ===== AI routes =====
	ai := r.Group("/ai-diagnoses")
	{
		ai.POST("", aidiagnosisHandler.Create)
		ai.GET("", aidiagnosisHandler.FindAllPending)
		ai.GET("/patient/:patient_id", aidiagnosisHandler.FindByPatientID)
		ai.PUT("/:id/verify", aidiagnosisHandler.Verify)
	}

	fr := r.Group("/full-records")
	{
		// 🩺 Tạo mới full record (record + attachment + prescription)
		fr.POST("/full", fullRecordHandler.CreateFullRecord)
		// 🩻 Hoàn thiện record đã có (thêm diagnosis, notes, attachments, prescription)
		fr.PUT("/complete", fullRecordHandler.CompleteRecord)
	}

	noti := r.Group("/notifications")
	{
		noti.POST("", handler.CreateNotification)
		noti.GET("", handler.GetAllNotifications)
		noti.PUT("/:id/read", handler.MarkNotificationRead)
		noti.PUT("/user/:userId/read-all", handler.MarkAllNotificationsRead)
		noti.DELETE("/:id", handler.DeleteNotification)
		noti.DELETE("/all", handler.DeleteAllNotifications)
		noti.GET("/unread", handler.CountUnreadNotifications)
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
