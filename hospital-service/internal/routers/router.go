package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/appointmenthandler"
	"hospital-service/internal/handlers/doctorhandler"
	"hospital-service/internal/handlers/drughandler"
	"hospital-service/internal/handlers/hospitalhandler"
	"hospital-service/internal/handlers/orderhandler"
	"hospital-service/internal/handlers/patienthandler"
	"hospital-service/internal/middlewares"

	"github.com/gin-contrib/cors"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler, doctorHandler *doctorhandler.DoctorHandler, hHandler *hospitalhandler.HospitalHandler, aHandler *appointmenthandler.AppointmentHandler, tHandler *appointmenthandler.TimeSlotHandler, drugHandler *drughandler.DrugHandler, orderHandler *orderhandler.OrderHandler) *gin.Engine {
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
		appointments.POST("", aHandler.CreateAppointment)
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
		timeSlot.GET("", tHandler.ListAllTimeSlots)
		timeSlot.GET("/:slot_id", tHandler.GetTimeSlotByID)
		timeSlot.GET("/doctor/:doctor_id", tHandler.GetTimeSlotsByDoctor)
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
		order.POST("", orderHandler.CreateOrder)                           // Create
		order.GET("", orderHandler.ListAllOrders)                          // List all
		order.GET("/:order_id", orderHandler.GetOrderByID)                 // Get by OrderID
		order.GET("/patient/:patient_id", orderHandler.GetOrdersByPatient) // Get orders by patient
		order.PUT("/:order_id/status", orderHandler.UpdateOrderStatus)     // Update order status
		order.PUT("/:order_id/detail", orderHandler.UpdateOrderDetail)     // Update order items/details
		order.DELETE("/:order_id", orderHandler.DeleteOrder)               // Delete order
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
