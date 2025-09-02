package routers

import (
	"hospital-service/internal/config"
	"hospital-service/internal/handlers/patienthandler"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, patientHandler *patienthandler.PatientHandler) *gin.Engine {
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

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
