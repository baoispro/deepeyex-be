package utils

type APIResponse struct {
	Status  int         `json:"status"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

// Helper functions (tuỳ chọn) để gọn code handler

func SuccessResponse(status int, message string, data interface{}) APIResponse {
	return APIResponse{
		Status:  status,
		Message: message,
		Data:    data,
	}
}

func ErrorResponse(status int, message string) APIResponse {
	return APIResponse{
		Status:  status,
		Message: message,
		Data:    nil,
	}
}
