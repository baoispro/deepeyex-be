package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/appointmenthandler"
	"hospital-service/internal/handlers/bookinghandler"
	"hospital-service/internal/handlers/doctorhandler"
	"hospital-service/internal/handlers/drughandler"
	"hospital-service/internal/handlers/hospitalhandler"
	"hospital-service/internal/handlers/medicalrecordhandler"
	"hospital-service/internal/handlers/orderhandler"
	"hospital-service/internal/handlers/patienthandler"
	"hospital-service/internal/handlers/paymenthandler"
	"hospital-service/internal/handlers/servicehandler"

	"hospital-service/internal/middlewares"

	"github.com/gin-contrib/cors"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler, doctorHandler *doctorhandler.DoctorHandler, hHandler *hospitalhandler.HospitalHandler, aHandler *appointmenthandler.AppointmentHandler, tHandler *appointmenthandler.TimeSlotHandler, drugHandler *drughandler.DrugHandler, orderHandler *orderhandler.OrderHandler, medicalRecordHandler *medicalrecordhandler.MedicalRecordHandler,
	prescriptionHandler *medicalrecordhandler.PrescriptionHandler,
	attachmentHandler *medicalrecordhandler.AttachmentHandler,
	followUpHandler *medicalrecordhandler.FollowUpHandler,
	prescriptionItemHandler *medicalrecordhandler.PrescriptionItemHandler,
	serviceHandler *servicehandler.ServiceHandler,
	bookingHandler *bookinghandler.BookingHandler,
	vnpayHandler *paymenthandler.VnpayHandler,
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
		appointments.PUT("/:appointment_id/status", aHandler.UpdateAppointmentStatus)
		appointments.PUT("/:appointment_id/detail", aHandler.UpdateAppointmentDetail)
		appointments.GET("", aHandler.ListAllAppointments)
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
		medical.POST("", medicalRecordHandler.CreateMedicalRecord)                    // Create
		medical.GET("", medicalRecordHandler.ListMedicalRecords)                      // List all
		medical.GET("/:record_id", medicalRecordHandler.GetMedicalRecord)             // Get by ID
		medical.GET("/:record_id/ai_diagnoses", medicalRecordHandler.ListAIDiagnoses) // List AI Diagnoses by MedicalRecord ID
		medical.PUT("/:record_id", medicalRecordHandler.UpdateMedicalRecord)          // Update
		medical.DELETE("/:record_id", medicalRecordHandler.DeleteMedicalRecord)
		medical.POST("/init", medicalRecordHandler.InitMedicalRecordAndDiagnosis)
		medical.POST("/:record_id/ai_diagnoses", medicalRecordHandler.AddAIDiagnosis)
		medical.GET("/ai_diagnoses/:id", medicalRecordHandler.GetAIDiagnosisByID)
		medical.DELETE("/ai_diagnoses/:id", medicalRecordHandler.DeleteAIDiagnosis)
		medical.GET("/ai_diagnoses/:id/recommended_plans", medicalRecordHandler.ListRecommendedPlans)
		medical.POST("/ai_diagnoses/:diagnosis_id/recommended_plans", medicalRecordHandler.AddRecommendedPlan)
		medical.DELETE("/ai_recommended_plans/:id", medicalRecordHandler.DeleteRecommendedPlan)

	}

	// ===== Prescription routes =====
	prescription := r.Group("/prescriptions")
	{
		prescription.POST("", prescriptionHandler.CreatePrescription)
		prescription.GET("/:prescription_id", prescriptionHandler.GetPrescriptionByID)
		prescription.GET("/medical_records/:record_id", prescriptionHandler.ListPrescriptionsByMedicalRecordID)
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

	// ===== FollowUp routes =====
	followup := r.Group("/followups")
	{
		followup.POST("/:record_id/medical_records", followUpHandler.CreateFollowUp)
		followup.GET("/:record_id/medical_records", followUpHandler.GetFollowUps)
		followup.PUT("/:follow_up_id", followUpHandler.UpdateFollowUp)
		followup.DELETE("/:follow_up_id", followUpHandler.DeleteFollowUp)
	}

	// ===== Service routes =====
	service := r.Group("/doctors/:doctor_id/services")
	{
		service.GET("", serviceHandler.ListServicesByDoctorID) // List services by Doctor ID
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

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
