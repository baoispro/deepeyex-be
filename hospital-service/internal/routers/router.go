package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/appointmenthandler"
	"hospital-service/internal/handlers/doctorhandler"
	"hospital-service/internal/handlers/hospitalhandler"
	"hospital-service/internal/handlers/patienthandler"
	"hospital-service/internal/middlewares"

	"github.com/gin-contrib/cors"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler, doctorHandler *doctorhandler.DoctorHandler, hHandler *hospitalhandler.HospitalHandler, aHandler *appointmenthandler.AppointmentHandler, tHandler *appointmenthandler.TimeSlotHandler) *gin.Engine {
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
		doctor.POST("", doctorHandler.CreateDoctor)                                 // Create
		doctor.GET("", doctorHandler.ListDoctors)                                   // List all
		doctor.GET("/user/:user_id", doctorHandler.GetDoctorByUserID)               // Get by UserID
		doctor.GET("/hospital/:hospital_id", doctorHandler.ListDoctorsByHospitalID) // List doctors by hospital_id
		doctor.GET("/:doctor_id", doctorHandler.GetDoctorByID)                      // Get by DoctorID
		doctor.PUT("/:doctor_id", doctorHandler.UpdateDoctor)                       // Update
		doctor.DELETE("/:doctor_id", doctorHandler.DeleteDoctor)                    // Delete
	}

	// ===== Hospital routes =====
	hospital := r.Group("/hospitals")
	{
		hospital.POST("", hHandler.CreateHospital)
		hospital.GET("", hHandler.ListHospitals)
		hospital.GET("/:hospital_id", hHandler.GetHospitalByID)
		hospital.PUT("/:hospital_id", hHandler.UpdateHospital)
		hospital.DELETE("/:hospital_id", hHandler.DeleteHospital)
	}

	appointments := r.Group("/appointments")
	{
		appointments.POST("", aHandler.CreateAppointment)
		appointments.GET("/:appointment_id", aHandler.GetAppointmentByID)
		appointments.GET("/patient/:patient_id", aHandler.GetAppointmentsByPatient)
		appointments.GET("/doctor/:doctor_id", aHandler.GetAppointmentsByDoctor)
		appointments.PUT("/:appointment_id/status", aHandler.UpdateAppointmentStatus)
		appointments.PUT("/:appointment_id/detail", aHandler.UpdateAppointmentDetail)
		appointments.GET("", aHandler.ListAllAppointments)
		appointments.DELETE("/:appointment_id", aHandler.DeleteAppointment)
	}

	timeSlot := r.Group("/timeslots")
	{
		timeSlot.POST("", tHandler.CreateTimeSlot)
		timeSlot.GET("", tHandler.ListAllTimeSlots)
		timeSlot.GET("/:slot_id", tHandler.GetTimeSlotByID)
		timeSlot.GET("/doctor/:doctor_id", tHandler.GetTimeSlotsByDoctor)
		timeSlot.PUT("/:slot_id", tHandler.UpdateTimeSlot)
		timeSlot.DELETE("/:slot_id", tHandler.DeleteTimeSlot)
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
