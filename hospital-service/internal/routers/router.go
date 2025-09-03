package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/doctorhandler"
	"hospital-service/internal/handlers/hospitalhandler"
	"hospital-service/internal/handlers/patienthandler"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler, doctorHandler *doctorhandler.DoctorHandler, hHandler *hospitalhandler.HospitalHandler) *gin.Engine {
	r := gin.Default()

	// ===== Patient routes =====
	patient := r.Group("/patients")
	{
		patient.POST("", patientHandler.CreatePatient)                     // Create
		patient.GET("", patientHandler.ListPatients)                        // List all
		patient.GET("/user/:user_id", patientHandler.GetPatientByUserID)   // Get by UserID
		patient.GET("/:patient_id", patientHandler.GetPatientByID)         // Get by PatientID
		patient.PUT("/:patient_id", patientHandler.UpdatePatient)          // Update
		patient.DELETE("/:patient_id", patientHandler.DeletePatient)       // Delete
	}

	// ===== Doctor routes =====
	doctor := r.Group("/doctors")
	{
		doctor.POST("", doctorHandler.CreateDoctor)                   // Create
		doctor.GET("", doctorHandler.ListDoctors)                     // List all
		doctor.GET("/user/:user_id", doctorHandler.GetDoctorByUserID) // Get by UserID
		doctor.GET("/hospital/:hospital_id", doctorHandler.ListDoctorsByHospitalID) // List doctors by hospital_id
		doctor.GET("/:doctor_id", doctorHandler.GetDoctorByID)        // Get by DoctorID
		doctor.PUT("/:doctor_id", doctorHandler.UpdateDoctor)         // Update
		doctor.DELETE("/:doctor_id", doctorHandler.DeleteDoctor)      // Delete
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


	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
