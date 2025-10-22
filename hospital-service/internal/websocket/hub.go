package websocket

import (
	"encoding/json"
	"log"
	"sync"
)

// MessageType định nghĩa loại message
type MessageType string

const (
	NewAppointment    MessageType = "NEW_APPOINTMENT"
	UpdateAppointment MessageType = "UPDATE_APPOINTMENT"
	CancelAppointment MessageType = "CANCEL_APPOINTMENT"
	NewNotification   MessageType = "NEW_NOTIFICATION"
)

// Message cấu trúc message gửi qua WebSocket
type Message struct {
	Type    MessageType `json:"type"`
	Payload interface{} `json:"payload"`
}

// Hub quản lý WebSocket connections
type Hub struct {
	// Clients được nhóm theo doctor_id
	// Key: doctor_id, Value: map of clients
	clients map[string]map[*Client]bool

	patientClients map[string]map[*Client]bool

	// Channel để broadcast message đến 1 doctor cụ thể
	broadcast        chan BroadcastMessage
	broadcastPatient chan BroadcastMessage

	// Channel để register client
	register chan *Client

	// Channel để unregister client
	unregister chan *Client

	// Mutex để đảm bảo thread-safe
	mu sync.RWMutex
}

// BroadcastMessage message broadcast đến doctor cụ thể
type BroadcastMessage struct {
	DoctorID  string
	PatientID string
	Message   Message
}

// NewHub tạo Hub mới
func NewHub() *Hub {
	return &Hub{
		clients:          make(map[string]map[*Client]bool),
		patientClients:   make(map[string]map[*Client]bool),
		broadcast:        make(chan BroadcastMessage, 256),
		broadcastPatient: make(chan BroadcastMessage, 256),
		register:         make(chan *Client),
		unregister:       make(chan *Client),
	}
}

// Run chạy hub, lắng nghe các channel
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.registerClient(client)

		case client := <-h.unregister:
			h.unregisterClient(client)

		case message := <-h.broadcast:
			h.broadcastToDoctor(message.DoctorID, message.Message)

		case msg := <-h.broadcastPatient:
			h.broadcastToPatient(msg.PatientID, msg.Message)
		}
	}
}

// registerClient đăng ký client mới
func (h *Hub) registerClient(client *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if client.DoctorID != "" {
		if h.clients[client.DoctorID] == nil {
			h.clients[client.DoctorID] = make(map[*Client]bool)
		}
		h.clients[client.DoctorID][client] = true
		log.Printf("[WebSocket] Doctor %s connected. Total: %d",
			client.DoctorID, len(h.clients[client.DoctorID]))
	}

	if client.PatientID != "" {
		if h.patientClients[client.PatientID] == nil {
			h.patientClients[client.PatientID] = make(map[*Client]bool)
		}
		h.patientClients[client.PatientID][client] = true
		log.Printf("[WebSocket] Patient %s connected. Total patient connections: %d",
			client.PatientID, len(h.patientClients[client.PatientID]))
	}
}

// unregisterClient hủy đăng ký client
func (h *Hub) unregisterClient(client *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	if client.DoctorID != "" {
		if clients, ok := h.clients[client.DoctorID]; ok {
			if _, exists := clients[client]; exists {
				delete(clients, client)
				close(client.send)

				// Nếu không còn client nào của doctor này, xóa map
				if len(clients) == 0 {
					delete(h.clients, client.DoctorID)
				}

				log.Printf("[WebSocket] Doctor %s disconnected. Remaining connections: %d",
					client.DoctorID, len(h.clients[client.DoctorID]))
			}
		}
	}

	if client.PatientID != "" {
		if clients, ok := h.patientClients[client.PatientID]; ok {
			delete(clients, client)
			close(client.send)
			if len(clients) == 0 {
				delete(h.patientClients, client.PatientID)
			}
			log.Printf("[WebSocket] Patient %s disconnected. Remaining: %d",
				client.PatientID, len(h.patientClients[client.PatientID]))
		}
	}
}

// broadcastToDoctor gửi message đến tất cả clients của 1 doctor
func (h *Hub) broadcastToDoctor(doctorID string, message Message) {
	h.mu.RLock()
	clients := h.clients[doctorID]
	h.mu.RUnlock()

	if clients == nil {
		log.Printf("[WebSocket] No active connections for doctor %s", doctorID)
		return
	}

	messageJSON, err := json.Marshal(message)
	if err != nil {
		log.Printf("[WebSocket] Error marshaling message: %v", err)
		return
	}

	log.Printf("[WebSocket] Broadcasting to doctor %s: %d clients", doctorID, len(clients))

	for client := range clients {
		select {
		case client.send <- messageJSON:
			// Message sent successfully
		default:
			// Client buffer full, close connection
			close(client.send)
			delete(clients, client)
		}
	}
}

// BroadcastToDoctor public method để broadcast từ bên ngoài
func (h *Hub) BroadcastToDoctor(doctorID string, messageType MessageType, payload interface{}) {
	message := Message{
		Type:    messageType,
		Payload: payload,
	}

	h.broadcast <- BroadcastMessage{
		DoctorID: doctorID,
		Message:  message,
	}
}

// GetConnectedDoctors trả về danh sách doctor IDs đang online
func (h *Hub) GetConnectedDoctors() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	doctors := make([]string, 0, len(h.clients))
	for doctorID := range h.clients {
		doctors = append(doctors, doctorID)
	}
	return doctors
}

// GetConnectionCount trả về số lượng connections của doctor
func (h *Hub) GetConnectionCount(doctorID string) int {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if clients, ok := h.clients[doctorID]; ok {
		return len(clients)
	}
	return 0
}

func (h *Hub) broadcastToPatient(patientID string, message Message) {
	h.mu.RLock()
	clients := h.patientClients[patientID]
	h.mu.RUnlock()

	if clients == nil {
		log.Printf("[WebSocket] No active patient connection for %s", patientID)
		return
	}

	data, _ := json.Marshal(message)
	for c := range clients {
		select {
		case c.send <- data:
		default:
			close(c.send)
			delete(clients, c)
		}
	}
}

func (h *Hub) BroadcastToPatient(patientID string, msgType MessageType, payload interface{}) {
	h.broadcastPatient <- BroadcastMessage{
		PatientID: patientID,
		Message: Message{
			Type:    msgType,
			Payload: payload,
		},
	}
}

func (h *Hub) GetConnectedPatients() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	patients := make([]string, 0, len(h.patientClients))
	for id := range h.patientClients {
		patients = append(patients, id)
	}
	return patients
}

func (h *Hub) GetConnectionCountPatient(patientID string) int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if clients, ok := h.patientClients[patientID]; ok {
		return len(clients)
	}
	return 0
}
