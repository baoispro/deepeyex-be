package websockethandler

import (
	"hospital-service/internal/utils"
	"hospital-service/internal/websocket"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	gorillaws "github.com/gorilla/websocket"
)

var upgrader = gorillaws.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// Allow connections from any origin (CORS)
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type WebSocketHandler struct {
	hub *websocket.Hub
}

func NewWebSocketHandler(hub *websocket.Hub) *WebSocketHandler {
	return &WebSocketHandler{hub: hub}
}

// ServeWS xử lý WebSocket connection cho bác sĩ
// @Summary WebSocket endpoint for doctors to receive real-time notifications
// @Description Doctors connect to this endpoint to receive real-time appointment notifications
// @Tags WebSocket
// @Param doctor_id query string true "Doctor ID"
// @Success 101 {string} string "Switching Protocols"
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /ws [get]
func (h *WebSocketHandler) ServeWS(c *gin.Context) {
	doctorID := c.Query("doctor_id")
	if doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "doctor_id is required"))
		return
	}

	// Upgrade HTTP connection to WebSocket
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("[WebSocket] Failed to upgrade connection for doctor %s: %v", doctorID, err)
		return
	}

	// Tạo client mới
	client := websocket.NewClient(h.hub, conn, doctorID)
	
	// Register client với hub (gửi qua channel)
	client.Register()

	// Start goroutines để handle read/write
	go client.WritePump()
	go client.ReadPump()

	log.Printf("[WebSocket] Doctor %s connected successfully", doctorID)
}

// GetConnectedDoctors trả về danh sách bác sĩ đang online
// @Summary Get list of connected doctors
// @Description Get list of doctor IDs currently connected via WebSocket
// @Tags WebSocket
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /ws/connected [get]
func (h *WebSocketHandler) GetConnectedDoctors(c *gin.Context) {
	doctors := h.hub.GetConnectedDoctors()
	
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Connected doctors retrieved", gin.H{
		"count":   len(doctors),
		"doctors": doctors,
	}))
}

// GetDoctorConnectionStatus kiểm tra trạng thái kết nối của bác sĩ
// @Summary Check doctor connection status
// @Description Check if a specific doctor is currently connected
// @Tags WebSocket
// @Param doctor_id path string true "Doctor ID"
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /ws/status/{doctor_id} [get]
func (h *WebSocketHandler) GetDoctorConnectionStatus(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	if doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "doctor_id is required"))
		return
	}

	connectionCount := h.hub.GetConnectionCount(doctorID)
	isConnected := connectionCount > 0

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Connection status retrieved", gin.H{
		"doctor_id":        doctorID,
		"is_connected":     isConnected,
		"connection_count": connectionCount,
	}))
}

//
// ====== PATIENT WEBSOCKET ======
//

// @Summary WebSocket endpoint for patients to receive real-time notifications
// @Description Patients connect to this endpoint to receive health or appointment updates
// @Tags WebSocket - Patient
// @Param patient_id query string true "Patient ID"
// @Success 101 {string} string "Switching Protocols"
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /ws/patient [get]
func (h *WebSocketHandler) ServeWSPatient(c *gin.Context) {
	patientID := c.Query("patient_id")
	if patientID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id is required"))
		return
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("[WebSocket] Failed to upgrade connection for patient %s: %v", patientID, err)
		return
	}

	client := websocket.NewClientPatient(h.hub, conn, patientID)
	client.Register()

	go client.WritePump()
	go client.ReadPump()

	log.Printf("[WebSocket] Patient %s connected successfully", patientID)
}

// @Summary Get list of connected patients
// @Tags WebSocket - Patient
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /ws/patient/connected [get]
func (h *WebSocketHandler) GetConnectedPatients(c *gin.Context) {
	patients := h.hub.GetConnectedPatients()

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Connected patients retrieved", gin.H{
		"count":    len(patients),
		"patients": patients,
	}))
}

// @Summary Check patient connection status
// @Tags WebSocket - Patient
// @Param patient_id path string true "Patient ID"
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /ws/patient/status/{patient_id} [get]
func (h *WebSocketHandler) GetPatientConnectionStatus(c *gin.Context) {
	patientID := c.Param("patient_id")
	if patientID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id is required"))
		return
	}

	connectionCount := h.hub.GetConnectionCountPatient(patientID)
	isConnected := connectionCount > 0

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Connection status retrieved", gin.H{
		"patient_id":       patientID,
		"is_connected":     isConnected,
		"connection_count": connectionCount,
	}))
}