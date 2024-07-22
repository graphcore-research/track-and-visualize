class SchemaException(Exception):
    """The provided DataFrame does not have a valid schema. See schema_migration and/or 
    LogFrame.schema to rectify the issue"""
    
    ...