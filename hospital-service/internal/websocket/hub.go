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

	// Channel để broadcast message đến 1 doctor cụ thể
	broadcast chan BroadcastMessage

	// Channel để register client
	register chan *Client

	// Channel để unregister client
	unregister chan *Client

	// Mutex để đảm bảo thread-safe
	mu sync.RWMutex
}

// BroadcastMessage message broadcast đến doctor cụ thể
type BroadcastMessage struct {
	DoctorID string
	Message  Message
}

// NewHub tạo Hub mới
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[string]map[*Client]bool),
		broadcast:  make(chan BroadcastMessage, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
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
		}
	}
}

// registerClient đăng ký client mới
func (h *Hub) registerClient(client *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.clients[client.DoctorID] == nil {
		h.clients[client.DoctorID] = make(map[*Client]bool)
	}
	h.clients[client.DoctorID][client] = true

	log.Printf("[WebSocket] Doctor %s connected. Total connections: %d", 
		client.DoctorID, len(h.clients[client.DoctorID]))
}

// unregisterClient hủy đăng ký client
func (h *Hub) unregisterClient(client *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()

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

