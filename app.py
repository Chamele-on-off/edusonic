@app.route('/api/room/initialize', methods=['POST'])
def force_room_initialization():
    """Принудительная инициализация комнаты"""
    try:
        data = request.json
        room_id = data.get('room_id')
        
        if not room_id:
            return jsonify({"success": False, "error": "Room ID is required"})
        
        success = initialize_room_safely(room_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Комната {room_id} инициализирована",
                "ready": check_room_ready(room_id),
                "retry_count": room_initialization_status[room_id]['retry_count']
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось инициализировать комнату {room_id}",
                "retry_count": room_initialization_status[room_id]['retry_count']
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/status/<room_id>')
def get_room_status(room_id):
    """Получение статуса комнаты"""
    try:
        status = {
            "room_id": room_id,
            "initialized": room_id in room_dialogue and room_dialogue[room_id] is not None,
            "participants": len(room_participants.get(room_id, [])),
            "ai_activated": room_ai_activated.get(room_id, False),
            "initialization_status": room_initialization_status.get(room_id, {}),
            "ready": check_room_ready(room_id)
        }
        
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/reset/<room_id>', methods=['POST'])
def reset_room(room_id):
    """Сброс состояния комнаты"""
    try:
        success = reset_room_state(room_id)
        return jsonify({
            "success": success,
            "message": f"Состояние комнаты {room_id} сброшено" if success else f"Ошибка сброса комнаты {room_id}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
