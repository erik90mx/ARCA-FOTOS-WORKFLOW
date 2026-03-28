#!/bin/bash
# Procesar grupos restantes de CBTIS 15 uno por uno

ARCA_DIR="$HOME/.openclaw/workspace/ARCA"
OUTPUT_DIR="$ARCA_DIR/2026/15/UNIDAS"

# Grupos restantes con nombres correctos
GROUPS=(
    "MAT_I_ELEC_DORADO"
    "MAT_J_MECIND_MARRON"
    "VESP_K_CONTA_PLATEADO"
    "VESP_L_PROG_GUINDA"
    "VESP_M_SYMEC_VERDE"
    "VESP_N_QUIM_BLANCO"
    "VESP_O_MECA_AZUL"
    "VESP_P_ELEC_DORADO"
    "VESP_Q_MECIND_MARRON"
)

cd "$ARCA_DIR"

for group in "${GROUPS[@]}"; do
    group_dir="$ARCA_DIR/2026/15/$group"
    
    if [ ! -d "$group_dir" ]; then
        echo "[SKIP] $group - directorio no existe"
        continue
    fi
    
    # Verificar si ya existe
    existing=$(ls "$OUTPUT_DIR"/*_${group}.webp 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        echo "[SKIP] $group - ya existe"
        continue
    fi
    
    echo "[PROC] Procesando $group..."
    
    # Ejecutar en proceso separado con timeout
    timeout 180 bash "$ARCA_DIR/arca_panorama.sh" 15 "$group" 2>&1 || {
        echo "[ERR] $group falló (exit code: $?)"
        continue
    }
    
    echo "[OK] $group completado"
    sleep 3  # Dar tiempo para liberar memoria
done

echo ""
echo "=== Resumen ==="
ls -1 "$OUTPUT_DIR"/*.webp 2>/dev/null | wc -l
echo "panorámicas totales"
