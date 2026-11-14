{{- define "algotr.name" -}}{{ .Release.Name }}{{- end -}}

{{- define "algotr.labels" -}}
app.kubernetes.io/name: algotr
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "algotr.secret" -}}
{{- if .Values.secrets.existingSecret -}}{{ .Values.secrets.existingSecret }}{{- else -}}{{ .Release.Name }}-credentials{{- end -}}
{{- end -}}

{{- define "algotr.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "algotr.env" -}}
- name: REDIS_URL
  value: {{ .Values.config.redisUrl | quote }}
- name: CELERY_BROKER_URL
  value: {{ .Values.config.celeryBrokerUrl | quote }}
- name: AUTO_FETCH_MODELS
  value: {{ .Values.config.autoFetchModels | quote }}
- name: ALPACA_PAPER
  value: {{ .Values.config.alpacaPaper | quote }}
- name: ALPACA_TRADING_MODE
  value: {{ .Values.live.tradingMode | quote }}
{{- if .Values.config.databaseUrl }}
- name: DATABASE_URL
  value: {{ .Values.config.databaseUrl | quote }}
{{- end }}
{{- end -}}

{{- define "algotr.volumes" -}}
{{- range $name := list "catalog" "events" "results" "data" }}
- name: {{ $name }}
  persistentVolumeClaim:
    claimName: {{ $.Release.Name }}-{{ $name }}
{{- end }}
{{- end -}}

{{- define "algotr.mounts" -}}
- {name: catalog, mountPath: /catalog}
- {name: events, mountPath: /events}
- {name: results, mountPath: /results}
- {name: data, mountPath: /app/data}
{{- end -}}
